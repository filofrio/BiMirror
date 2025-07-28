// BiMirror_Parallel.cpp : Versione ottimizzata con parallelizzazione OpenMP
//

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

class TrigonometricLUT {
private:
   std::vector<double> sin_values;
   std::vector<double> cos_values;
   double angle_step;
   std::string lut_filename;

public:
   TrigonometricLUT(double step = 0.5, const std::string& filename = "trigonometric_lut.dat")
      : angle_step(step), lut_filename(filename) {
   }

   void precomputeAndSave() {
      std::cout << "Precalcolo valori trigonometrici..." << std::endl;

      sin_values.clear();
      cos_values.clear();

      int num_angles = static_cast<int>(180.0 / angle_step);
      sin_values.resize(num_angles);
      cos_values.resize(num_angles);

      // Parallelizza il precalcolo trigonometrico
#pragma omp parallel for
      for (int i = 0; i < num_angles; i++) {
         double angle_deg = i * angle_step;
         double angle_rad = angle_deg * CV_PI / 180.0;
         sin_values[i] = std::sin(angle_rad);
         cos_values[i] = std::cos(angle_rad);
      }

      // Salva in file binario
      std::ofstream file(lut_filename, std::ios::binary);
      if (!file.is_open()) {
         throw std::runtime_error("Impossibile creare il file LUT: " + lut_filename);
      }

      file.write(reinterpret_cast<const char*>(&angle_step), sizeof(angle_step));
      size_t size = sin_values.size();
      file.write(reinterpret_cast<const char*>(&size), sizeof(size));
      file.write(reinterpret_cast<const char*>(sin_values.data()), size * sizeof(double));
      file.write(reinterpret_cast<const char*>(cos_values.data()), size * sizeof(double));

      file.close();
      std::cout << "LUT trigonometrica salvata: " << num_angles << " angoli precalcolati" << std::endl;
   }

   bool loadFromFile() {
      std::ifstream file(lut_filename, std::ios::binary);
      if (!file.is_open()) {
         return false;
      }

      double saved_step;
      file.read(reinterpret_cast<char*>(&saved_step), sizeof(saved_step));

      if (std::abs(saved_step - angle_step) > 1e-9) {
         file.close();
         return false;
      }

      size_t size;
      file.read(reinterpret_cast<char*>(&size), sizeof(size));

      sin_values.resize(size);
      cos_values.resize(size);

      file.read(reinterpret_cast<char*>(sin_values.data()), size * sizeof(double));
      file.read(reinterpret_cast<char*>(cos_values.data()), size * sizeof(double));

      file.close();
      std::cout << "LUT trigonometrica caricata: " << size << " angoli" << std::endl;
      return true;
   }

   void initialize() {
      if (!loadFromFile()) {
         std::cout << "File LUT non trovato o non compatibile, precalcolo..." << std::endl;
         precomputeAndSave();
      }
   }

   std::pair<double, double> getSinCos(double angle_deg) const {
      if (sin_values.empty()) {
         throw std::runtime_error("LUT non inizializzata");
      }

      double index_exact = angle_deg / angle_step;
      int index = static_cast<int>(index_exact);

      if (index < 0) index = 0;
      if (index >= static_cast<int>(sin_values.size())) index = static_cast<int>(sin_values.size()) - 1;

      return { sin_values[index], cos_values[index] };
   }
};

float calcolaAngoloLineaDuePunti(const cv::Vec4f& linea) {
   // Estrai i due punti
   float x1 = linea[0];
   float y1 = linea[1];
   float x2 = linea[2];
   float y2 = linea[3];

   // Calcola il vettore direzione
   double dx = x2 - x1;
   double dy = y2 - y1;

   // Calcola l'angolo usando atan2
   double angolo_rad = std::atan2(dy, dx);

   // Converti da radianti a gradi
   double angolo_gradi = angolo_rad * 180.0 / CV_PI;

   // Normalizza l'angolo nell'intervallo [-90, +90]
   if (angolo_gradi > 90.0) {
      angolo_gradi -= 180.0;
   }
   else if (angolo_gradi < -90.0) {
      angolo_gradi += 180.0;
   }

   return static_cast<float>(angolo_gradi);
}

// per normalizzare angoli di elementi simmetrici
double normalizzaAngolo(double angolo) {
   // Normalizza prima in [-180, 180)
   while (angolo >= 180.0) angolo -= 360.0;
   while (angolo < -180.0) angolo += 360.0;

   // Poi riduci a [-90, 90]
   if (angolo > 90.0) {
      angolo -= 180.0;
   }
   else if (angolo < -90.0) {
      angolo += 180.0;
   }

   return angolo;
}

class BiMirrorLensAnalyzer {
private:
   cv::Rect roiLente;
   cv::Mat original_image;
   cv::Mat processed_image;
   cv::Mat roi_image;
   cv::Mat lens_mask;
   cv::Point2f lens_center;
   cv::Point2f lens_center_global;
   float lens_radius;
   double optimal_angle;
   cv::Vec4f axis_line;
   cv::Vec4f axis_line_global;
   TrigonometricLUT trig_lut;
   

public:
  
   // Enum per le metriche di valutazione
   enum class EvaluationMetric {
      INTEGRAL_VALUE = 0,    // Valore dell'integrale
      INTEGRAL_AMPLITUDE = 1, // Ampiezza della striscia chiara
      COMBINED_SCORE = 2,    // Score combinato (integrale × √ampiezza)
      MAX_INTEGRAL = 3       // Valore massimo dell'integrale
   };

   struct IntegralResult {
      double angle;
      double integral_value;
      double integral_amplitude;
      double combined_score;
      double max_integral;   // NUOVO: valore massimo dell'integrale
      std::vector<double> profile;
      std::vector<int> positions;
      int bright_stripe_start;
      int bright_stripe_end;
   };

   struct BandLineResult {
      double slope;           // Pendenza della retta (m in y = mx + q)
      double intercept;       // Intercetta (q in y = mx + q)
      double distance;        // Distanza dal centro della lente
      cv::Point2f closest_point; // Punto sulla retta più vicino al centro
      double r_squared;       // Coefficiente di determinazione (qualità del fit)
      std::vector<cv::Point2f> band_points; // Punti rilevati della banda
      cv::Vec4f fitted_line;  // Linea per la visualizzazione
   };

   BandLineResult band_analysis;

   BiMirrorLensAnalyzer() : lens_radius(0), optimal_angle(0), trig_lut(0.5) {
#ifdef _OPENMP
      std::cout << "OpenMP disponibile - Thread disponibili: " << omp_get_max_threads() << std::endl;
#else
      std::cout << "OpenMP non disponibile - Esecuzione sequenziale" << std::endl;
#endif
   }

   void initializeTrigonometry() {
      trig_lut.initialize();
   }

   cv::Point2f roiToGlobal(const cv::Point2f& roiPoint) {
      return cv::Point2f(roiPoint.x + roiLente.x, roiPoint.y + roiLente.y);
   }

   cv::Point2f globalToRoi(const cv::Point2f& globalPoint) {
      return cv::Point2f(globalPoint.x - roiLente.x, globalPoint.y - roiLente.y);
   }

   void loadImage(const std::string& filepath) {
      try {
         original_image = cv::imread(filepath, cv::IMREAD_COLOR);
         if (original_image.empty()) {
            throw std::runtime_error("Impossibile caricare l'immagine: " + filepath);
         }

         cv::cvtColor(original_image, processed_image, cv::COLOR_BGR2GRAY);

         roiLente.x = 250;
         roiLente.y = 50;
         roiLente.width = 700;
         roiLente.height = 600;

         if (roiLente.x + roiLente.width > processed_image.cols ||
            roiLente.y + roiLente.height > processed_image.rows) {
            throw std::runtime_error("ROI fuori dai limiti dell'immagine");
         }

         roi_image = processed_image(roiLente);
      }
      catch (const cv::Exception& e) {
         throw std::runtime_error("Errore OpenCV durante il caricamento: " + std::string(e.what()));
      }
   }

   void detectLens() {
      try {
         cv::Mat blurred;
         cv::GaussianBlur(roi_image, blurred, cv::Size(9, 9), 2);

         std::vector<cv::Vec3f> circles;
         cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT, 1,
            roi_image.rows / 4,
            70, 5,
            roi_image.rows / 4,
            roi_image.rows / 2);

         if (circles.empty()) {
            throw std::runtime_error("Nessuna lente circolare rilevata nell'immagine");
         }

         int max_radius_idx = 0;
         for (size_t i = 1; i < circles.size(); i++) {
            if (circles[i][2] > circles[max_radius_idx][2]) {
               max_radius_idx = static_cast<int>(i);
            }
         }

         lens_center = cv::Point2f(circles[max_radius_idx][0], circles[max_radius_idx][1]);
         lens_radius = circles[max_radius_idx][2];
         lens_center_global = roiToGlobal(lens_center);

         lens_mask = cv::Mat::zeros(roi_image.size(), CV_8UC1);
         cv::circle(lens_mask, lens_center, static_cast<int>(lens_radius), cv::Scalar(255), -1);

      }
      catch (const cv::Exception& e) {
         throw std::runtime_error("Errore durante il rilevamento della lente: " + std::string(e.what()));
      }
   }

   // Versione ottimizzata con parallelizzazione
   IntegralResult analyzeRotationProfile(double rotation_angle) {
      try {
         IntegralResult result;
         result.angle = rotation_angle;

         auto [sin_val, cos_val] = trig_lut.getSinCos(rotation_angle);

         cv::Mat rotation_matrix = cv::getRotationMatrix2D(lens_center, rotation_angle, 1.0);
         cv::Mat rotated, rotated_mask;
         cv::warpAffine(roi_image, rotated, rotation_matrix, roi_image.size());
         cv::warpAffine(lens_mask, rotated_mask, rotation_matrix, lens_mask.size());

         int start_y = std::max(0, static_cast<int>(lens_center.y - lens_radius));
         int end_y = std::min(rotated.rows, static_cast<int>(lens_center.y + lens_radius));
         int num_rows = end_y - start_y;

         if (num_rows < 10) {
            result.integral_value = 0;
            result.integral_amplitude = 0;
            result.combined_score = 0;
            result.max_integral = 0;  // NUOVO
            return result;
         }

         result.profile.resize(num_rows);
         result.positions.resize(num_rows);

         // Parallelizza il calcolo del profilo riga per riga
#pragma omp parallel for
         for (int idx = 0; idx < num_rows; idx++) {
            int y = start_y + idx;
            double row_sum = 0;
            int pixel_count = 0;

            for (int x = 0; x < rotated.cols; x++) {
               if (rotated_mask.at<uchar>(y, x) > 0) {
                  row_sum += rotated.at<uchar>(y, x);
                  pixel_count++;
               }
            }

            if (pixel_count > 0) {
               result.profile[idx] = row_sum / pixel_count;
               result.positions[idx] = y;
            }
            else {
               result.profile[idx] = 0;
               result.positions[idx] = y;
            }
         }

         calculateIntegralMetrics(result);
         return result;

      }
      catch (const std::exception& e) {
         throw std::runtime_error("Errore durante l'analisi rotazione: " + std::string(e.what()));
      }
   }

   void calculateIntegralMetrics(IntegralResult& result) {
      if (result.profile.empty()) return;

      double min_val = *std::min_element(result.profile.begin(), result.profile.end());
      double max_val = *std::max_element(result.profile.begin(), result.profile.end());

      if (max_val == min_val) {
         result.integral_value = 0;
         result.integral_amplitude = 0;
         result.combined_score = 0;
         result.max_integral = 0;  // NUOVO
         return;
      }

      std::vector<double> normalized_profile(result.profile.size());

      // Parallelizza la normalizzazione
      int profile_size = static_cast<int>(result.profile.size());
#pragma omp parallel for
      for (int i = 0; i < profile_size; i++) {
         normalized_profile[i] = (result.profile[i] - min_val) / (max_val - min_val);
      }

      // Calcola la media (operazione di riduzione)
      double profile_mean = 0;
#pragma omp parallel for reduction(+:profile_mean)
      for (int i = 0; i < profile_size; i++) {
         profile_mean += normalized_profile[i];
      }
      profile_mean /= normalized_profile.size();

      result.integral_value = 0;
      std::vector<int> bright_regions;

      // Calcola integrale e identifica regioni chiare
      for (size_t i = 0; i < normalized_profile.size(); i++) {
         if (normalized_profile[i] > profile_mean) {
            result.integral_value += (normalized_profile[i] - profile_mean);
            bright_regions.push_back(static_cast<int>(i));
         }
      }

      // NUOVO: Calcola il valore massimo dell'integrale nel profilo
      result.max_integral = 0;
      if (!normalized_profile.empty()) {
         result.max_integral = *std::max_element(normalized_profile.begin(), normalized_profile.end());
      }

      if (!bright_regions.empty()) {
         std::vector<std::pair<int, int>> continuous_regions;
         int start = bright_regions[0];
         int end = bright_regions[0];

         for (size_t i = 1; i < bright_regions.size(); i++) {
            if (bright_regions[i] == bright_regions[i - 1] + 1) {
               end = bright_regions[i];
            }
            else {
               continuous_regions.push_back({ start, end });
               start = bright_regions[i];
               end = bright_regions[i];
            }
         }
         continuous_regions.push_back({ start, end });

         int max_width = 0;
         int best_start = 0, best_end = 0;
         for (auto& region : continuous_regions) {
            int width = region.second - region.first + 1;
            if (width > max_width) {
               max_width = width;
               best_start = region.first;
               best_end = region.second;
            }
         }

         result.integral_amplitude = max_width;
         result.bright_stripe_start = best_start;
         result.bright_stripe_end = best_end;
      }
      else {
         result.integral_amplitude = 0;
         result.bright_stripe_start = -1;
         result.bright_stripe_end = -1;
      }

      result.combined_score = result.integral_value * std::sqrt(result.integral_amplitude);
   }

   void createAnalysisGraphs(const std::vector<IntegralResult>& results) {
      const int graph_width = 800;
      const int graph_height = 600;
      const int margin = 60;
      const int plot_width = graph_width - 2 * margin;
      const int plot_height = graph_height - 2 * margin;

      double max_integral = 0, max_amplitude = 0, max_score = 0;
      for (const auto& result : results) {
         max_integral = std::max(max_integral, result.integral_value);
         max_amplitude = std::max(max_amplitude, result.integral_amplitude);
         max_score = std::max(max_score, result.combined_score);
      }

      // Grafico 1: Integrale vs Angolo
      cv::Mat graph1(graph_height, graph_width, CV_8UC3, cv::Scalar(255, 255, 255));
      drawGraphFrame(graph1, "Integrale vs Angolo", "Angolo (gradi)", "Valore Integrale", margin);

      for (size_t i = 1; i < results.size(); i++) {
         int x1 = margin + static_cast<int>((results[i - 1].angle / 180.0) * plot_width);
         int y1 = graph_height - margin - static_cast<int>((results[i - 1].integral_value / max_integral) * plot_height);
         int x2 = margin + static_cast<int>((results[i].angle / 180.0) * plot_width);
         int y2 = graph_height - margin - static_cast<int>((results[i].integral_value / max_integral) * plot_height);
         cv::line(graph1, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
      }

      // Grafico 2: Ampiezza vs Angolo
      cv::Mat graph2(graph_height, graph_width, CV_8UC3, cv::Scalar(255, 255, 255));
      drawGraphFrame(graph2, "Ampiezza vs Angolo", "Angolo (gradi)", "Ampiezza", margin);

      for (size_t i = 1; i < results.size(); i++) {
         int x1 = margin + static_cast<int>((results[i - 1].angle / 180.0) * plot_width);
         int y1 = graph_height - margin - static_cast<int>((results[i - 1].integral_amplitude / max_amplitude) * plot_height);
         int x2 = margin + static_cast<int>((results[i].angle / 180.0) * plot_width);
         int y2 = graph_height - margin - static_cast<int>((results[i].integral_amplitude / max_amplitude) * plot_height);
         cv::line(graph2, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
      }

      // Grafico 3: Score Combinato vs Angolo
      cv::Mat graph3(graph_height, graph_width, CV_8UC3, cv::Scalar(255, 255, 255));
      drawGraphFrame(graph3, "Score Combinato vs Angolo", "Angolo (gradi)", "Score", margin);

      for (size_t i = 1; i < results.size(); i++) {
         int x1 = margin + static_cast<int>((results[i - 1].angle / 180.0) * plot_width);
         int y1 = graph_height - margin - static_cast<int>((results[i - 1].combined_score / max_score) * plot_height);
         int x2 = margin + static_cast<int>((results[i].angle / 180.0) * plot_width);
         int y2 = graph_height - margin - static_cast<int>((results[i].combined_score / max_score) * plot_height);
         cv::line(graph3, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
      }

      int optimal_x = margin + static_cast<int>((optimal_angle / 180.0) * plot_width);
      cv::line(graph1, cv::Point(optimal_x, margin), cv::Point(optimal_x, graph_height - margin), cv::Scalar(0, 255, 255), 2);
      cv::line(graph2, cv::Point(optimal_x, margin), cv::Point(optimal_x, graph_height - margin), cv::Scalar(0, 255, 255), 2);
      cv::line(graph3, cv::Point(optimal_x, margin), cv::Point(optimal_x, graph_height - margin), cv::Scalar(0, 255, 255), 2);

      cv::imshow("Grafico 1: Integrale vs Angolo", graph1);
      cv::imshow("Grafico 2: Ampiezza vs Angolo", graph2);
      cv::imshow("Grafico 3: Score Combinato vs Angolo", graph3);
   }

   void drawGraphFrame(cv::Mat& graph, const std::string& title, const std::string& xlabel,
      const std::string& ylabel, int margin) {
      cv::rectangle(graph, cv::Point(margin, margin),
         cv::Point(graph.cols - margin, graph.rows - margin), cv::Scalar(0, 0, 0), 2);

      cv::putText(graph, title, cv::Point(graph.cols / 2 - static_cast<int>(title.length()) * 5, 30),
         cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

      cv::putText(graph, xlabel, cv::Point(graph.cols / 2 - static_cast<int>(xlabel.length()) * 4, graph.rows - 10),
         cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

      cv::putText(graph, ylabel, cv::Point(10, graph.rows / 2),
         cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

      for (int angle = 0; angle <= 180; angle += 30) {
         int x = margin + static_cast<int>((angle / 180.0) * (graph.cols - 2 * margin));
         cv::line(graph, cv::Point(x, graph.rows - margin),
            cv::Point(x, graph.rows - margin + 5), cv::Scalar(0, 0, 0), 1);
         cv::putText(graph, std::to_string(angle), cv::Point(x - 10, graph.rows - margin + 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
      }
   }

   // Funzione helper per ottenere il valore della metrica scelta
   double getMetricValue(const IntegralResult& result, EvaluationMetric metric) {
      switch (metric) {
      case EvaluationMetric::INTEGRAL_VALUE:
         return result.integral_value;
      case EvaluationMetric::INTEGRAL_AMPLITUDE:
         return result.integral_amplitude;
      case EvaluationMetric::COMBINED_SCORE:
         return result.combined_score;
      case EvaluationMetric::MAX_INTEGRAL:
         return result.max_integral;
      default:
         return result.combined_score; // Default fallback
      }
   }

   // Funzione helper per ottenere il nome della metrica
   std::string getMetricName(EvaluationMetric metric) {
      switch (metric) {
      case EvaluationMetric::INTEGRAL_VALUE:
         return "Valore Integrale";
      case EvaluationMetric::INTEGRAL_AMPLITUDE:
         return "Ampiezza Striscia";
      case EvaluationMetric::COMBINED_SCORE:
         return "Score Combinato";
      case EvaluationMetric::MAX_INTEGRAL:
         return "Valore Massimo Integrale";
      default:
         return "Score Combinato";
      }
   }

   // LOOP PRINCIPALE PARALLELIZZATO CON SCELTA METRICA
   std::vector<IntegralResult> findOptimalOrientation(EvaluationMetric metric = EvaluationMetric::COMBINED_SCORE) {
      try {
         std::cout << "Inizio analisi sistematica rotazioni (versione parallela)..." << std::endl;
         std::cout << "Metrica di valutazione: " << getMetricName(metric) << std::endl;

         auto start_time = std::chrono::high_resolution_clock::now();

         double angle_step = 0.5;
         double max_angle = 180.0;
         int num_angles = static_cast<int>(max_angle / angle_step);

         std::vector<IntegralResult> results(num_angles);

         // PARALLELIZZAZIONE DEL LOOP PRINCIPALE
#pragma omp parallel for schedule(dynamic, 4)
         for (int i = 0; i < num_angles; i++) {
            double angle = i * angle_step;
            results[i] = analyzeRotationProfile(angle);

            // Progress report thread-safe
            if (i % 20 == 0) {
#pragma omp critical
               {
                  std::cout << "   Processato angolo: " << angle << "° (thread: "
                     << omp_get_thread_num() << ", metrica: "
                     << getMetricValue(results[i], metric) << ")" << std::endl;
               }
            }
         }

         auto end_time = std::chrono::high_resolution_clock::now();
         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
         std::cout << "Tempo di elaborazione: " << duration.count() << " ms" << std::endl;

         // Trova il risultato migliore basato sulla metrica scelta
         auto best_result = std::max_element(results.begin(), results.end(),
            [this, metric](const IntegralResult& a, const IntegralResult& b) {
               return getMetricValue(a, metric) < getMetricValue(b, metric);
            });

         if (best_result == results.end() || getMetricValue(*best_result, metric) == 0) {
            throw std::runtime_error("Impossibile trovare orientamento ottimale con la metrica: " + getMetricName(metric));
         }

         optimal_angle = best_result->angle;

         std::cout << "\n=== RISULTATO OTTIMALE ===" << std::endl;
         std::cout << "Metrica utilizzata: " << getMetricName(metric) << std::endl;
         std::cout << "Angolo ottimale trovato: " << optimal_angle << "°    " << normalizzaAngolo(optimal_angle) << "°" << std::endl;
         std::cout << "Valore metrica ottimale: " << getMetricValue(*best_result, metric) << std::endl;
         std::cout << "\n--- Dettagli completi ---" << std::endl;
         std::cout << "Integrale: " << best_result->integral_value << std::endl;
         std::cout << "Ampiezza striscia: " << best_result->integral_amplitude << std::endl;
         std::cout << "Score combinato: " << best_result->combined_score << std::endl;
         std::cout << "Valore max integrale: " << best_result->max_integral << std::endl;

         return results;

      }
      catch (const std::exception& e) {
         throw std::runtime_error("Errore durante la ricerca dell'orientamento: " + std::string(e.what()));
      }
   }

   void calculateOptimalCentralAxis() {
      try {
         IntegralResult optimal_result = analyzeRotationProfile(optimal_angle);

         if (optimal_result.bright_stripe_start == -1 || optimal_result.bright_stripe_end == -1) {
            throw std::runtime_error("Impossibile determinare i limiti della striscia chiara");
         }

         int center_idx = (optimal_result.bright_stripe_start + optimal_result.bright_stripe_end) / 2;
         int center_y = optimal_result.positions[center_idx];

         cv::Point2f p1_rotated(lens_center.x - lens_radius, static_cast<float>(center_y));
         cv::Point2f p2_rotated(lens_center.x + lens_radius, static_cast<float>(center_y));

         cv::Mat inv_rotation = cv::getRotationMatrix2D(lens_center, -optimal_angle, 1.0);

         std::vector<cv::Point2f> points_rotated = { p1_rotated, p2_rotated };
         std::vector<cv::Point2f> points_roi;
         cv::transform(points_rotated, points_roi, inv_rotation);

         axis_line = cv::Vec4f(points_roi[0].x, points_roi[0].y, points_roi[1].x, points_roi[1].y);

         cv::Point2f p1_global = roiToGlobal(points_roi[0]);
         cv::Point2f p2_global = roiToGlobal(points_roi[1]);

         axis_line_global = cv::Vec4f(p1_global.x, p1_global.y, p2_global.x, p2_global.y);
         float angolo = calcolaAngoloLineaDuePunti(axis_line_global);
         std::cout << "Angolo ottimale trovato con altra variabile: " << angolo << "°" << std::endl;

      }
      catch (const cv::Exception& e) {
         throw std::runtime_error("Errore nel calcolo dell'asse ottimale: " + std::string(e.what()));
      }
   }

   void analyze(EvaluationMetric metric = EvaluationMetric::COMBINED_SCORE) {
      try {
         std::cout << "=== ANALISI LENTE BI-MIRROR (Versione Parallela con OpenMP) ===" << std::endl;

         std::cout << "\n0. Inizializzazione LUT trigonometrica..." << std::endl;
         initializeTrigonometry();

         std::cout << "\n1. Rilevamento lente nel ROI..." << std::endl;
         std::cout << "   ROI: x=" << roiLente.x << ", y=" << roiLente.y
            << ", w=" << roiLente.width << ", h=" << roiLente.height << std::endl;
         detectLens();
         std::cout << "   Lente trovata:" << std::endl;
         std::cout << "     - Centro nel ROI: (" << lens_center.x << ", " << lens_center.y << ")" << std::endl;
         std::cout << "     - Centro globale: (" << lens_center_global.x << ", "
            << lens_center_global.y << ")" << std::endl;
         std::cout << "     - Raggio: " << lens_radius << std::endl;

         std::cout << "\n2. Ricerca orientamento ottimale (parallela)..." << std::endl;
         std::vector<IntegralResult> results = findOptimalOrientation(metric);

         std::cout << "\n3. Calcolo asse centrale ottimale..." << std::endl;
         calculateOptimalCentralAxis();
         std::cout << "   Asse centrale calcolato con successo!" << std::endl;

         // Analisi della posizione reale della banda
         std::cout << "\n3.5 Analisi posizione reale della banda chiara..." << std::endl;
         band_analysis = analyzeBandPosition();

         std::cout << "\n4. Generazione grafici di analisi..." << std::endl;
         createAnalysisGraphs(results);
         std::cout << "   Grafici generati e visualizzati!" << std::endl;

      }
      catch (const std::exception& e) {
         throw std::runtime_error("Errore durante l'analisi: " + std::string(e.what()));
      }
   }

   void visualizeResults(const std::string& window_name = "Risultati Analisi Bi Mirror - Parallela") {
      try {
         cv::Mat display;
         original_image.copyTo(display);

         // Visualizzazioni esistenti
         cv::rectangle(display, roiLente, cv::Scalar(255, 255, 0), 2);
         cv::circle(display, lens_center_global, static_cast<int>(lens_radius), cv::Scalar(0, 255, 0), 2);

         // Linea teorica (passante per il centro) - in rosso
         cv::line(display,
            cv::Point(static_cast<int>(axis_line_global[0]), static_cast<int>(axis_line_global[1])),
            cv::Point(static_cast<int>(axis_line_global[2]), static_cast<int>(axis_line_global[3])),
            cv::Scalar(0, 0, 255), 2);

         // Linea reale della banda - in blu
         cv::Point2f p1_roi(band_analysis.fitted_line[0], band_analysis.fitted_line[1]);
         cv::Point2f p2_roi(band_analysis.fitted_line[2], band_analysis.fitted_line[3]);
         cv::Point2f p1_global = roiToGlobal(p1_roi);
         cv::Point2f p2_global = roiToGlobal(p2_roi);

         cv::line(display,
            cv::Point(static_cast<int>(p1_global.x), static_cast<int>(p1_global.y)),
            cv::Point(static_cast<int>(p2_global.x), static_cast<int>(p2_global.y)),
            cv::Scalar(255, 0, 0), 2);

         // Linea di distanza minima - in verde
         cv::Point2f closest_global = roiToGlobal(band_analysis.closest_point);
         cv::line(display,
            cv::Point(static_cast<int>(lens_center_global.x), static_cast<int>(lens_center_global.y)),
            cv::Point(static_cast<int>(closest_global.x), static_cast<int>(closest_global.y)),
            cv::Scalar(0, 255, 0), 1);

         // Testi informativi
         cv::putText(display, "ROI",
            cv::Point(roiLente.x + 5, roiLente.y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

         cv::putText(display, "Linea Teorica (rosso) vs Reale (blu)",
            cv::Point(50, 50),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

         std::string angle_text = "Angolo: " + std::to_string(normalizzaAngolo(optimal_angle)) + " gradi";
         cv::putText(display, angle_text,
            cv::Point(50, 80),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

         std::string distance_text = "Distanza dal centro: " +
            std::to_string(static_cast<int>(band_analysis.distance)) + " pixel";
         cv::putText(display, distance_text,
            cv::Point(50, 110),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

         cv::imshow(window_name, display);
         cv::waitKey(0);

      }
      catch (const cv::Exception& e) {
         throw std::runtime_error("Errore nella visualizzazione: " + std::string(e.what()));
      }
   }

   void saveResults(const std::string& output_path) {
      try {
         cv::Mat output;
         original_image.copyTo(output);

         cv::rectangle(output, roiLente, cv::Scalar(255, 255, 0), 2);
         cv::circle(output, lens_center_global, static_cast<int>(lens_radius), cv::Scalar(0, 255, 0), 2);
         cv::line(output,
            cv::Point(static_cast<int>(axis_line_global[0]), static_cast<int>(axis_line_global[1])),
            cv::Point(static_cast<int>(axis_line_global[2]), static_cast<int>(axis_line_global[3])),
            cv::Scalar(0, 0, 255), 3);

         if (!cv::imwrite(output_path, output)) {
            throw std::runtime_error("Impossibile salvare l'immagine");
         }

         std::string txt_path = output_path.substr(0, output_path.find_last_of('.')) + "_params_parallel.txt";
         std::ofstream params_file(txt_path);

         if (!params_file.is_open()) {
            throw std::runtime_error("Impossibile creare il file dei parametri");
         }

         params_file << "Parametri Analisi Lente Bi Mirror - Versione Parallela OpenMP\n";
         params_file << "=============================================================\n";
         params_file << "Metodo: Rotazione sistematica ogni 0.5 gradi con LUT e parallelizzazione\n";
#ifdef _OPENMP
         params_file << "Thread utilizzati: " << omp_get_max_threads() << "\n";
#else
         params_file << "OpenMP: Non disponibile\n";
#endif
         params_file << "ROI: x=" << roiLente.x << ", y=" << roiLente.y
            << ", width=" << roiLente.width << ", height=" << roiLente.height << "\n";
         params_file << "Centro lente (ROI): (" << lens_center.x << ", " << lens_center.y << ")\n";
         params_file << "Centro lente (globale): (" << lens_center_global.x << ", "
            << lens_center_global.y << ")\n";
         params_file << "Raggio lente: " << lens_radius << "\n";
         params_file << "Angolo ottimale: " << optimal_angle << " gradi\n";
         params_file << "Asse centrale (ROI): (" << axis_line[0] << ", " << axis_line[1]
            << ") -> (" << axis_line[2] << ", " << axis_line[3] << ")\n";
         params_file << "Asse centrale (globale): (" << axis_line_global[0] << ", " << axis_line_global[1]
            << ") -> (" << axis_line_global[2] << ", " << axis_line_global[3] << ")\n";

         params_file << "\n=== Analisi Posizione Banda Chiara ===\n";
         params_file << "Distanza dal centro: " << band_analysis.distance << " pixel\n";
         params_file << "Qualità regressione (R²): " << band_analysis.r_squared << "\n";
         params_file << "Punti rilevati: " << band_analysis.band_points.size() << "\n";
         params_file << "Posizione banda: " <<
            (band_analysis.closest_point.y < lens_center.y ? "sopra" : "sotto") << " il centro\n";
         params_file << "Linea banda reale (ROI): (" << band_analysis.fitted_line[0] << ", "
            << band_analysis.fitted_line[1] << ") -> ("
            << band_analysis.fitted_line[2] << ", " << band_analysis.fitted_line[3] << ")\n";

         params_file.close();

      }
      catch (const std::exception& e) {
         throw std::runtime_error("Errore nel salvataggio: " + std::string(e.what()));
      }
   }
   BandLineResult performLinearRegression(const std::vector<cv::Point2f>& points) {
      BandLineResult result;

      if (points.size() < 2) {
         throw std::runtime_error("Insufficienti punti per la regressione lineare");
      }

      // Calcola le medie
      double sum_x = 0, sum_y = 0;
      for (const auto& p : points) {
         sum_x += p.x;
         sum_y += p.y;
      }
      double mean_x = sum_x / points.size();
      double mean_y = sum_y / points.size();

      // Calcola i coefficienti della regressione
      double sum_xx = 0, sum_xy = 0, sum_yy = 0;
      for (const auto& p : points) {
         double dx = p.x - mean_x;
         double dy = p.y - mean_y;
         sum_xx += dx * dx;
         sum_xy += dx * dy;
         sum_yy += dy * dy;
      }

      // Gestisce il caso di linea verticale
      if (std::abs(sum_xx) < 1e-10) {
         // Linea verticale: x = costante
         result.slope = std::numeric_limits<double>::infinity();
         result.intercept = mean_x;
      }
      else {
         // Linea normale: y = mx + q
         result.slope = sum_xy / sum_xx;
         result.intercept = mean_y - result.slope * mean_x;
      }

      // Calcola R²
      if (sum_yy > 0) {
         double ss_res = 0;
         for (const auto& p : points) {
            double y_pred = result.slope * p.x + result.intercept;
            double residual = p.y - y_pred;
            ss_res += residual * residual;
         }
         result.r_squared = 1.0 - (ss_res / sum_yy);
      }
      else {
         result.r_squared = 1.0; // Perfetto fit orizzontale
      }

      return result;
   }

   // Metodo per calcolare la distanza punto-retta
   double pointToLineDistance(const cv::Point2f& point, double slope, double intercept, cv::Point2d& closest_point) {
      if (std::isinf(slope)) {
         // Linea verticale: x = intercept
         closest_point.x = static_cast<float>(intercept);
         closest_point.y = static_cast<float>(point.y);
         return std::abs(point.x - intercept);
      }
      else {
         // Linea normale: ax + by + c = 0 dove a = -slope, b = 1, c = -intercept
         double a = -slope;
         double b = 1.0;
         double c = -intercept;

         // Distanza = |ax + by + c| / sqrt(a² + b²)
         double distance = std::abs(a * point.x + b * point.y + c) / std::sqrt(a * a + b * b);

         // Trova il punto più vicino sulla retta
         double factor = -(a * point.x + b * point.y + c) / (a * a + b * b);
         closest_point.x = point.x + a * factor;
         closest_point.y = point.y + b * factor;

         return distance;
      }
   }

   // Nuovo metodo principale per analizzare la posizione della banda
   BandLineResult analyzeBandPosition() {
      std::cout << "\nAnalisi posizione reale della banda chiara..." << std::endl;

      // Usa l'angolo ottimale già trovato
      auto [sin_val, cos_val] = trig_lut.getSinCos(optimal_angle);

      // Ruota l'immagine all'angolo ottimale
      cv::Mat rotation_matrix = cv::getRotationMatrix2D(lens_center, optimal_angle, 1.0);
      cv::Mat rotated, rotated_mask;
      cv::warpAffine(roi_image, rotated, rotation_matrix, roi_image.size());
      cv::warpAffine(lens_mask, rotated_mask, rotation_matrix, lens_mask.size());

      // Trova i punti di massima intensità per ogni colonna
      std::vector<cv::Point2f> band_points;

      int start_x = std::max(0, static_cast<int>(lens_center.x - lens_radius));
      int end_x = std::min(rotated.cols, static_cast<int>(lens_center.x + lens_radius));

		for (int x = start_x; x < end_x; x += 1) { // per cambiare il passo di campionamento modifica x+= 1
         // Determina il range y per questa colonna
         double dx = x - lens_center.x;
         double max_dy = std::sqrt(lens_radius * lens_radius - dx * dx);

         int y_start = std::max(0, static_cast<int>(lens_center.y - max_dy));
         int y_end = std::min(rotated.rows, static_cast<int>(lens_center.y + max_dy));

         // Trova il picco di intensità in questa colonna
         int max_intensity = 0;
         int best_y = -1;

         for (int y = y_start; y < y_end; y++) {
            if (rotated_mask.at<uchar>(y, x) > 0) {
               int intensity = rotated.at<uchar>(y, x);
               if (intensity > max_intensity) {
                  max_intensity = intensity;
                  best_y = y;
               }
            }
         }

         if (best_y != -1 && max_intensity > 100) { // Soglia minima di intensità
            band_points.push_back(cv::Point2f(static_cast<float>(x), static_cast<float>(best_y)));
         }
      }

      if (band_points.size() < 10) {
         throw std::runtime_error("Punti insufficienti per determinare la banda chiara");
      }

      // Esegui la regressione lineare
      BandLineResult result = performLinearRegression(band_points);
      result.band_points = band_points;

      // Calcola la distanza dal centro
      cv::Point2d closest_point;
      result.distance = pointToLineDistance(lens_center, result.slope, result.intercept, closest_point);
      result.closest_point = closest_point;

      // Crea la linea per la visualizzazione
      if (std::isinf(result.slope)) {
         // Linea verticale
         result.fitted_line = cv::Vec4f(static_cast<float>(result.intercept), 0, static_cast<float>(result.intercept), static_cast<float>(rotated.rows));
      }
      else {
         // Calcola i punti estremi della linea
         float y1 = static_cast<float>(result.slope * start_x + result.intercept);
         float y2 = static_cast<float>(result.slope * end_x + result.intercept);
         result.fitted_line = cv::Vec4f(static_cast<float>(start_x), y1, static_cast<float>(end_x), y2);
      }

      // Trasforma i risultati nelle coordinate originali (non ruotate)
      cv::Mat inv_rotation = cv::getRotationMatrix2D(lens_center, -optimal_angle, 1.0);
      std::vector<cv::Point2f> line_points = {
          cv::Point2f(result.fitted_line[0], result.fitted_line[1]),
          cv::Point2f(result.fitted_line[2], result.fitted_line[3]),
          result.closest_point
      };
      std::vector<cv::Point2f> transformed_points;
      cv::transform(line_points, transformed_points, inv_rotation);

      // Aggiorna la linea nelle coordinate originali
      result.fitted_line = cv::Vec4f(
         transformed_points[0].x, transformed_points[0].y,
         transformed_points[1].x, transformed_points[1].y
      );
      result.closest_point = transformed_points[2];

      // Output dei risultati
      std::cout << "   Punti banda rilevati: " << band_points.size() << std::endl;
      std::cout << "   Qualità del fit (R²): " << result.r_squared << std::endl;
      std::cout << "   Distanza dal centro: " << result.distance << " pixel" << std::endl;
      std::cout << "   Posizione banda: " << (closest_point.y < lens_center.y ? "sopra" : "sotto") << " il centro" << std::endl;

      return result;
   }

};

int main(int argc, char** argv) {
   try {
#ifdef _OPENMP
      // Configura il numero di thread OpenMP (opzionale)
      // omp_set_num_threads(4); // Forza 4 thread se necessario
#endif

      std::string nomeFileImg = "imgBiDeg.png";
      BiMirrorLensAnalyzer::EvaluationMetric metrica = BiMirrorLensAnalyzer::EvaluationMetric::INTEGRAL_VALUE;

      // Parsing argomenti
      if (argc >= 2) {
         nomeFileImg = argv[1];
      }

      // Opzione per scegliere la metrica (4° parametro)
      if (argc >= 4) {
         int metric_choice = std::atoi(argv[3]);
         switch (metric_choice) {
         case 0:
            metrica = BiMirrorLensAnalyzer::EvaluationMetric::INTEGRAL_VALUE;
            std::cout << "Metrica selezionata: Valore Integrale" << std::endl;
            break;
         case 1:
            metrica = BiMirrorLensAnalyzer::EvaluationMetric::INTEGRAL_AMPLITUDE;
            std::cout << "Metrica selezionata: Ampiezza Striscia" << std::endl;
            break;
         case 2:
            metrica = BiMirrorLensAnalyzer::EvaluationMetric::COMBINED_SCORE;
            std::cout << "Metrica selezionata: Score Combinato" << std::endl;
            break;
         case 3:
            metrica = BiMirrorLensAnalyzer::EvaluationMetric::MAX_INTEGRAL;
            std::cout << "Metrica selezionata: Valore Massimo Integrale" << std::endl;
            break;
         default:
            std::cout << "Metrica non valida, uso Score Combinato (default)" << std::endl;
            metrica = BiMirrorLensAnalyzer::EvaluationMetric::COMBINED_SCORE;
            break;
         }
      }
      else {
         std::cout << "Metrica di default: Score Combinato" << std::endl;
         std::cout << "Per scegliere una metrica diversa: ./programma immagine.png output.png METRICA" << std::endl;
         std::cout << "   METRICA: 0=Integrale, 1=Ampiezza, 2=ScoreCombinato, 3=MaxIntegrale" << std::endl;
      }

      BiMirrorLensAnalyzer analyzer;

      std::cout << "\nCaricamento immagine: " << nomeFileImg << std::endl;
      analyzer.loadImage(nomeFileImg);

      std::cout << "\nAvvio analisi lente Bi Mirror (versione parallela)...\n" << std::endl;
      analyzer.analyze(metrica);

      std::cout << "\nVisualizzazione risultati..." << std::endl;
      analyzer.visualizeResults();

      if (argc >= 3) {
         std::cout << "\nSalvataggio risultati in: " << argv[2] << std::endl;
         analyzer.saveResults(argv[2]);
         std::cout << "Risultati salvati con successo!" << std::endl;
      }

      std::cout << "\nPremi un tasto per chiudere i grafici..." << std::endl;
      cv::waitKey(0);
      cv::destroyAllWindows();

      return 0;

   }
   catch (const std::exception& e) {
      std::cerr << "\nERRORE: " << e.what() << std::endl;
      std::cerr << "\nUSO: " << std::endl;
      std::cerr << "  " << (argv ? argv[0] : "programma") << " <immagine.png> [output.png] [metrica]" << std::endl;
      std::cerr << "  metrica: 0=Integrale, 1=Ampiezza, 2=ScoreCombinato(default), 3=MaxIntegrale" << std::endl;
      return 1;
   }
}