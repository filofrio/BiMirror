﻿// BiMirror_Parallel.cpp : Versione ottimizzata con parallelizzazione OpenMP
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
#include <numeric>
#include <immintrin.h>  // Per AVX/SSE
#include <future>
#include <opencv2/core/ocl.hpp>


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

// questa classe per il caching delle rotazioni
class RotationCache {
private:
   struct CacheEntry {
      cv::Mat rotated_image;
      cv::Mat rotated_mask;
   };
   std::unordered_map<int, CacheEntry> cache;

public:
   bool getRotation(double angle, cv::Mat& rotated, cv::Mat& mask) {
      int angle_key = static_cast<int>(angle * 2); // Per step di 0.5
      auto it = cache.find(angle_key);
      if (it != cache.end()) {
         rotated = it->second.rotated_image;
         mask = it->second.rotated_mask;
         return true;
      }
      return false;
   }

   void storeRotation(double angle, const cv::Mat& rotated, const cv::Mat& mask) {
      int angle_key = static_cast<int>(angle * 2);
      cache[angle_key] = { rotated.clone(), mask.clone() };
   }
};



class BiMirrorLensAnalyzer {
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
   EvaluationMetric current_metric;  // Memorizza la metrica corrente
   IntegralResult optimal_result;
   bool has_optimal_result;
   cv::Mat temp_rotated;
   cv::Mat temp_rotated_mask;
   std::vector<double> temp_profile;
   std::vector<int> temp_positions;

   // Pool di thread per operazioni asincrone
   std::vector<std::future<IntegralResult>> futures;
   std::vector<IntegralResult> processAngleBatchGPU(const std::vector<double>& angles, cv::UMat& gpu_roi, cv::UMat& gpu_mask)
   {
      std::vector<IntegralResult> results(angles.size());
      std::vector<std::future<IntegralResult>> futures;

      //// Step 1: Esegui tutte le rotazioni del batch
      //std::vector<cv::UMat> rotated_images(angles.size());
      //std::vector<cv::UMat> rotated_masks(angles.size());

      //// Rotazioni in parallelo su GPU
      //for (size_t i = 0; i < angles.size(); i++) {
      //   cv::Mat rotation_matrix = cv::getRotationMatrix2D(lens_center, angles[i], 1.0);
      //   cv::warpAffine(gpu_roi, rotated_images[i], rotation_matrix, roi_image.size());
      //   cv::warpAffine(gpu_mask, rotated_masks[i], rotation_matrix, lens_mask.size());
      //}

      //// Step 2: Analizza ogni immagine ruotata con reduce
      //for (size_t i = 0; i < angles.size(); i++) {
      //   results[i] = analyzeRotatedImageGPU(rotated_images[i], rotated_masks[i], angles[i]);
      //}

      // Lancia le rotazioni in modo asincrono
      for (size_t i = 0; i < angles.size(); i++) {
         futures.push_back(
            std::async(std::launch::async, [this, &gpu_roi, &gpu_mask, angles, i]() {
               cv::UMat rotated, mask;
               cv::Mat M = cv::getRotationMatrix2D(lens_center, angles[i], 1.0);
               cv::warpAffine(gpu_roi, rotated, M, roi_image.size());
               cv::warpAffine(gpu_mask, mask, M, lens_mask.size());
               return analyzeRotatedImageGPU(rotated, mask, angles[i]);
               })
         );
      }

      // Raccogli i risultati
      for (size_t i = 0; i < futures.size(); i++) {
         results[i] = futures[i].get();
      }

      return results;
   }

   // Analisi ottimizzata con reduce operations
   IntegralResult analyzeRotatedImageGPU(
      cv::UMat& rotated,
      cv::UMat& mask,
      double angle)
   {
      IntegralResult result;
      result.angle = angle;

      int start_y = std::max(0, static_cast<int>(lens_center.y - lens_radius));
      int end_y = std::min(rotated.rows, static_cast<int>(lens_center.y + lens_radius));
      int num_rows = end_y - start_y;

      if (num_rows < 10) {
         result.integral_value = 0;
         result.integral_amplitude = 0;
         result.combined_score = 0;
         result.max_integral = 0;
         return result;
      }

      // Estrai la striscia
      cv::UMat strip = rotated(cv::Range(start_y, end_y), cv::Range::all());
      cv::UMat mask_strip = mask(cv::Range(start_y, end_y), cv::Range::all());

      // IMPORTANTE: Usa CV_64F per mantenere la precisione double!
      cv::UMat strip_double, mask_double;
      strip.convertTo(strip_double, CV_64F);

      // Crea una maschera binaria (0 o 1) invece di 0-255
      cv::UMat mask_binary;
      cv::threshold(mask_strip, mask_binary, 0, 1.0, cv::THRESH_BINARY);
      mask_binary.convertTo(mask_double, CV_64F);

      // Applica la maschera
      cv::UMat masked;
      cv::multiply(strip_double, mask_double, masked);

      // Somma i valori e conta i pixel validi
      cv::UMat row_sums, pixel_counts;
      cv::reduce(masked, row_sums, 1, cv::REDUCE_SUM, CV_64F);
      cv::reduce(mask_double, pixel_counts, 1, cv::REDUCE_SUM, CV_64F);

      // Scarica dalla GPU
      cv::Mat sums_cpu, counts_cpu;
      row_sums.copyTo(sums_cpu);
      pixel_counts.copyTo(counts_cpu);

      // Calcola il profilo
      result.profile.resize(num_rows);
      result.positions.resize(num_rows);

      for (int i = 0; i < num_rows; i++) {
         double pixel_count = counts_cpu.at<double>(i);
         if (pixel_count > 0) {
            result.profile[i] = sums_cpu.at<double>(i) / pixel_count;
         }
         else {
            result.profile[i] = 0;
         }
         result.positions[i] = start_y + i;
      }

      calculateIntegralMetrics(result);
      return result;
   }
public:
   BiMirrorLensAnalyzer() : lens_radius(0), optimal_angle(0), trig_lut(0.5), has_optimal_result(false) {
#ifdef _OPENMP
      std::cout << "OpenMP disponibile - Thread disponibili: " << omp_get_max_threads() << std::endl;
#else
      std::cout << "OpenMP non disponibile - Esecuzione sequenziale" << std::endl;
#endif
      temp_profile.reserve(1000);
      temp_positions.reserve(1000);
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
         original_image = cv::imread(filepath, cv::IMREAD_ANYCOLOR);
         if (original_image.empty()) {
            throw std::runtime_error("Impossibile caricare l'immagine: " + filepath);
         }

         cv::cvtColor(original_image, processed_image, cv::COLOR_BGR2GRAY);

         roiLente.x = 250;
         roiLente.y = 50;
         roiLente.width = 700;
         roiLente.height = 600;
         //roiLente.x = 50;
         //roiLente.y = 10;
         //roiLente.width = 7000;
         //roiLente.height = 6000;

         if(roiLente.x + roiLente.width > processed_image.cols)
				roiLente.width = processed_image.cols - roiLente.x - 1;
         if (roiLente.y + roiLente.height > processed_image.rows)
            roiLente.height = processed_image.rows - roiLente.y - 1;

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
         cv::GaussianBlur(roi_image, blurred, cv::Size(9, 9), 2); // ho scoperto che ci vuole...basta uno spot di rumore per far fallire i gradienti per i circle di Hough

         std::vector<cv::Vec3f> circles;
         cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT, 1,
         //cv::HoughCircles(roi_image, circles, cv::HOUGH_GRADIENT, 1,
            roi_image.rows / 4,
				30, 30,     // 50, 5 era prima         50,11       50,12    primo par e' la soglia superiore di Canny, secondo par la sensibilita' per i contorni ... piu' basso maggior sensibilita'
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
            result.max_integral = 0;
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

   IntegralResult analyzeRotationProfileGPU(double rotation_angle) {
      try {
         IntegralResult result;
         result.angle = rotation_angle;

         auto [sin_val, cos_val] = trig_lut.getSinCos(rotation_angle);

         // Usa UMat per operazioni GPU
         cv::UMat roi_umat, mask_umat;
         roi_image.copyTo(roi_umat);
         lens_mask.copyTo(mask_umat);

         cv::UMat rotated_umat, rotated_mask_umat;
         cv::Mat rotation_matrix = cv::getRotationMatrix2D(lens_center, rotation_angle, 1.0);

         // Queste operazioni sono accelerate via OpenCL
         cv::warpAffine(roi_umat, rotated_umat, rotation_matrix, roi_image.size(), cv::INTER_LINEAR);
         cv::warpAffine(mask_umat, rotated_mask_umat, rotation_matrix, lens_mask.size(), cv::INTER_LINEAR);

         // Converti back a Mat per l'analisi del profilo
         cv::Mat rotated, rotated_mask;
         rotated_umat.copyTo(rotated);
         rotated_mask_umat.copyTo(rotated_mask);

         int start_y = std::max(0, static_cast<int>(lens_center.y - lens_radius));
         int end_y = std::min(rotated.rows, static_cast<int>(lens_center.y + lens_radius));
         int num_rows = end_y - start_y;

         if (num_rows < 10) {
            result.integral_value = 0;
            result.integral_amplitude = 0;
            result.combined_score = 0;
            result.max_integral = 0;
            return result;
         }

         result.profile.resize(num_rows);
         result.positions.resize(num_rows);

         // Il calcolo del profilo resta su CPU (difficile da parallelizzare su GPU per questa operazione)
#pragma omp parallel for
         for (int idx = 0; idx < num_rows; idx++) {
            int y = start_y + idx;
            double row_sum = 0;
            int pixel_count = 0;

            const uchar* mask_row = rotated_mask.ptr<uchar>(y);
            const uchar* img_row = rotated.ptr<uchar>(y);

            for (int x = 0; x < rotated.cols; x++) {
               if (mask_row[x] > 0) {
                  row_sum += img_row[x];
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
         // Fallback su versione CPU in caso di errore
         std::cerr << "Errore GPU, fallback su CPU: " << e.what() << std::endl;
         return analyzeRotationProfile(rotation_angle);
      }
   }

   std::vector<IntegralResult> analyzeRotationBatchGPU(const std::vector<double>& angles) {
      std::vector<IntegralResult> results(angles.size());

      // Pre-carica una sola volta
      cv::UMat roi_umat, mask_umat;
      roi_image.copyTo(roi_umat);
      lens_mask.copyTo(mask_umat);

      // Buffer per il batch
      std::vector<cv::UMat> rotated_batch(angles.size());
      std::vector<cv::UMat> mask_batch(angles.size());

      // FASE 1: Tutte le rotazioni su GPU in parallelo
      for (size_t i = 0; i < angles.size(); i++) {
         cv::Mat M = cv::getRotationMatrix2D(lens_center, angles[i], 1.0);
         cv::warpAffine(roi_umat, rotated_batch[i], M, roi_image.size());
         cv::warpAffine(mask_umat, mask_batch[i], M, lens_mask.size());
      }

      // FASE 2: Analisi con reduce (opzione 3) per ogni immagine ruotata
      for (size_t i = 0; i < angles.size(); i++) {
         // Calcola profilo con reduce operations GPU-ottimizzate
         results[i] = calculateProfileWithReduce(rotated_batch[i], mask_batch[i], angles[i]);
      }

      return results;
   }

   IntegralResult calculateProfileWithReduce(cv::UMat& rotated, cv::UMat& mask, double angle) {
      IntegralResult result;
      result.angle = angle;

      // Estrai la striscia
      int start_y = std::max(0, static_cast<int>(lens_center.y - lens_radius));
      int end_y = std::min(rotated.rows, static_cast<int>(lens_center.y + lens_radius));

      cv::UMat strip = rotated(cv::Range(start_y, end_y), cv::Range::all());
      cv::UMat mask_strip = mask(cv::Range(start_y, end_y), cv::Range::all());

      // Converti a float
      cv::UMat strip_float, mask_float;
      strip.convertTo(strip_float, CV_32F);
      mask_strip.convertTo(mask_float, CV_32F, 1.0 / 255.0);

      // Moltiplica per la maschera
      cv::UMat masked;
      cv::multiply(strip_float, mask_float, masked);

      // Reduce per sommare lungo le righe (MOLTO veloce su GPU)
      cv::UMat row_sums, row_counts;
      cv::reduce(masked, row_sums, 1, cv::REDUCE_SUM);
      cv::reduce(mask_float, row_counts, 1, cv::REDUCE_SUM);

      // Una sola copia GPU->CPU per batch
      cv::Mat sums, counts;
      row_sums.copyTo(sums);
      row_counts.copyTo(counts);

      // Calcolo veloce delle medie
      result.profile.resize(sums.rows);
      result.positions.resize(sums.rows);

      for (int i = 0; i < sums.rows; i++) {
         float count = counts.at<float>(i);
         result.profile[i] = (count > 0) ? sums.at<float>(i) / count : 0;
         result.positions[i] = start_y + i;
      }

      calculateIntegralMetrics(result);
      return result;
   }

   void enableOpenCVOptimizations() {
      // Abilita OpenCL se disponibile
      cv::ocl::setUseOpenCL(true);

      // Verifica se è attivo
      if (cv::ocl::useOpenCL()) {
         std::cout << "OpenCL abilitato per accelerazione GPU" << std::endl;

         // Mostra info sul dispositivo
         std::vector<cv::ocl::PlatformInfo> platforms;
         cv::ocl::getPlatfomsInfo(platforms);

         for (size_t i = 0; i < platforms.size(); i++) {
            const cv::ocl::PlatformInfo& platform = platforms[i];
            std::cout << "  Piattaforma " << i << ": " << platform.name() << std::endl;

            for (int j = 0; j < platform.deviceNumber(); j++) {
               cv::ocl::Device device;
               platform.getDevice(device, j);
               std::cout << "    Device: " << device.name() << std::endl;
               std::cout << "    Tipo: " << (device.type() == cv::ocl::Device::TYPE_GPU ? "GPU" : "CPU") << std::endl;
            }
         }

         // Pre-carica i kernel OpenCL con operazione dummy
         cv::UMat temp;
         roi_image.copyTo(temp);
         cv::UMat rotated;
         cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(static_cast<float>(temp.cols) / 2.f, static_cast<float>(temp.rows) / 2.f), 45, 1.0);
         cv::warpAffine(temp, rotated, M, temp.size());
         std::cout << "  Kernel OpenCL pre-caricati" << std::endl;
      }
      else {
         std::cout << "OpenCL non disponibile - uso ottimizzazioni CPU" << std::endl;
      }

      // Abilita anche ottimizzazioni CPU
      cv::setNumThreads(cv::getNumberOfCPUs());
      cv::setUseOptimized(true);

      std::cout << "Ottimizzazioni CPU: " << cv::getNumberOfCPUs() << " thread" << std::endl;
      std::cout << "SIMD: " << (cv::useOptimized() ? "Abilitato" : "Disabilitato") << std::endl;
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
   std::vector<IntegralResult> findOptimalOrientation(EvaluationMetric metric = EvaluationMetric::INTEGRAL_VALUE) {
      try {
         std::cout << "Inizio analisi sistematica rotazioni (versione parallela)..." << std::endl;
         std::cout << "Metrica di valutazione: " << getMetricName(metric) << std::endl;

         bool use_gpu = cv::ocl::useOpenCL();
         if (use_gpu) {
            std::cout << "Uso accelerazione GPU OpenCL per le rotazioni" << std::endl;
         }
         else {
            std::cout << "OpenCL non disponibile, uso CPU" << std::endl;
         }

         auto start_time = std::chrono::high_resolution_clock::now();

         double angle_step = 0.5;
         double max_angle = 180.0;
         int num_angles = static_cast<int>(max_angle / angle_step);

         std::vector<IntegralResult> results(num_angles);

         // PARALLELIZZAZIONE DEL LOOP PRINCIPALE
#pragma omp parallel for schedule(dynamic, 4)
         for (int i = 0; i < num_angles; i++) {
            double angle = i * angle_step;

            if (use_gpu) {
               results[i] = analyzeRotationProfileGPU(angle);
            }
            else {
               results[i] = analyzeRotationProfile(angle);
            }

            // Progress report thread-safe
            if (i % 40 == 0) {
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

         // NUOVO: Salva il risultato completo, non solo l'angolo
         optimal_result = *best_result;
         optimal_angle = best_result->angle;
         has_optimal_result = true;

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

   std::vector<IntegralResult> findOptimalOrientationBatchGPU(EvaluationMetric metric = EvaluationMetric::INTEGRAL_VALUE)
   {
      try {
         std::cout << "Inizio analisi con Batch GPU Processing + Reduce Operations..." << std::endl;
         std::cout << "Metrica di valutazione: " << getMetricName(metric) << std::endl;

         // Verifica disponibilità GPU
         if (!cv::ocl::useOpenCL()) {
            std::cout << "GPU non disponibile, uso metodo CPU standard" << std::endl;
            return findOptimalOrientation(metric);
         }

         auto start_time = std::chrono::high_resolution_clock::now();

         const double angle_step = 0.5;
         const double max_angle = 180.0;
         const int num_angles = static_cast<int>(max_angle / angle_step);
         const int BATCH_SIZE = 32; // Dimensione ottimale del batch

         // Pre-carica i dati sulla GPU UNA SOLA VOLTA
         cv::UMat gpu_roi, gpu_mask;
         roi_image.copyTo(gpu_roi);
         lens_mask.copyTo(gpu_mask);
         std::cout << "Dati caricati su GPU" << std::endl;

         std::vector<IntegralResult> all_results;
         all_results.reserve(num_angles);

         // Processa in batch
         for (int batch_start = 0; batch_start < num_angles; batch_start += BATCH_SIZE) {
            int batch_end = std::min(batch_start + BATCH_SIZE, num_angles);
            int current_batch_size = batch_end - batch_start;

            // Prepara gli angoli per questo batch
            std::vector<double> batch_angles;
            batch_angles.reserve(current_batch_size);
            for (int i = batch_start; i < batch_end; i++) {
               batch_angles.push_back(i * angle_step);
            }

            // Processa il batch
            auto batch_results = processAngleBatchGPU(batch_angles, gpu_roi, gpu_mask);

            // Aggiungi i risultati
            all_results.insert(all_results.end(),
               batch_results.begin(),
               batch_results.end());

            // Progress report
            int progress = (batch_end * 100) / num_angles;
            std::cout << "\rProgresso GPU Batch: " << progress << "% "
               << "(Batch " << (batch_start / BATCH_SIZE + 1)
               << "/" << ((num_angles + BATCH_SIZE - 1) / BATCH_SIZE) << ")"
               << std::flush;
         }
         std::cout << std::endl;

         auto end_time = std::chrono::high_resolution_clock::now();
         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
         std::cout << "Tempo totale elaborazione GPU: " << duration.count() << " ms" << std::endl;
         std::cout << "Tempo medio per angolo: "
            << (duration.count() / static_cast<double>(num_angles)) << " ms" << std::endl;

         // Trova il risultato ottimale
         auto best_result = std::max_element(all_results.begin(), all_results.end(),
            [this, metric](const IntegralResult& a, const IntegralResult& b) {
               return getMetricValue(a, metric) < getMetricValue(b, metric);
            });

         if (best_result == all_results.end() || getMetricValue(*best_result, metric) == 0) {
            throw std::runtime_error("Impossibile trovare orientamento ottimale");
         }

         // Salva i risultati
         optimal_result = *best_result;
         optimal_angle = best_result->angle;
         has_optimal_result = true;

         std::cout << "\n=== RISULTATO OTTIMALE (GPU Batch) ===" << std::endl;
         std::cout << "Angolo ottimale: " << optimal_angle << "°" << std::endl;
         std::cout << "Valore metrica: " << getMetricValue(*best_result, metric) << std::endl;

         return all_results;

      }
      catch (const cv::Exception& e) {
         std::cerr << "Errore GPU: " << e.what() << std::endl;
         std::cout << "Fallback su metodo CPU standard" << std::endl;
         return findOptimalOrientation(metric);
      }
   }

   void calculateOptimalCentralAxis() {
      try {
         // Verifica che abbiamo un risultato ottimale salvato
         if (!has_optimal_result) {
            throw std::runtime_error("Nessun risultato ottimale disponibile. Eseguire prima findOptimalOrientation()");
         }

         // Usa direttamente i dati salvati invece di ricalcolare
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

   void analyze(EvaluationMetric metric = EvaluationMetric::INTEGRAL_VALUE) {

      current_metric = metric;  // Salva la metrica per uso successivo
      try {
         std::cout << "=== ANALISI LENTE BI-MIRROR (Versione Parallela con OpenMP) ===" << std::endl;

         static bool opencv_optimized = false;
         if (!opencv_optimized) {
            std::cout << "\n-1. Inizializzazione ottimizzazioni OpenCV..." << std::endl;
            enableOpenCVOptimizations();
            opencv_optimized = true;
         }

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

         std::vector<IntegralResult> results;

         // Usa il metodo GPU batch se disponibile
         if (cv::ocl::useOpenCL()) {
            results = findOptimalOrientationBatchGPU(metric);
         }
         else {
            results = findOptimalOrientation(metric);
         }

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

         // Calcola il fattore di scala ottimale per la visualizzazione
         double scale_factor = 1.;
         const int max_display_width = 1200;   // Larghezza massima desiderata
         const int max_display_height = 800;   // Altezza massima desiderata

         // Calcola il fattore di scala necessario
         double scale_x = 1.0;
         double scale_y = 1.0;

         if (display.cols > max_display_width) {
            scale_x = static_cast<double>(max_display_width) / display.cols;
         }
         if (display.rows > max_display_height) {
            scale_y = static_cast<double>(max_display_height) / display.rows;
         }

         // Usa il fattore di scala minore per mantenere le proporzioni
         scale_factor = std::min(scale_x, scale_y);

         // Applica lo zoom solo se necessario
         cv::Mat display_scaled;
         cv::Mat display_for_drawing;

         if (scale_factor < 1.0) {
            // Ridimensiona l'immagine
            cv::resize(display, display_scaled, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
            display_for_drawing = display_scaled;

            std::cout << "Immagine ridimensionata con fattore: " << scale_factor
               << " (da " << display.cols << "x" << display.rows
               << " a " << display_scaled.cols << "x" << display_scaled.rows << ")" << std::endl;
         }
         else {
            display_for_drawing = display;
            std::cout << "Nessun ridimensionamento necessario" << std::endl;
         }

         // Scala tutti i punti e le coordinate per il disegno
         cv::Rect roiLente_scaled(
            static_cast<int>(roiLente.x * scale_factor),
            static_cast<int>(roiLente.y * scale_factor),
            static_cast<int>(roiLente.width * scale_factor),
            static_cast<int>(roiLente.height * scale_factor)
         );

         cv::Point lens_center_global_scaled(
            static_cast<int>(lens_center_global.x * scale_factor),
            static_cast<int>(lens_center_global.y * scale_factor)
         );

         int lens_radius_scaled = static_cast<int>(lens_radius * scale_factor);

         // Disegna gli elementi scalati
         cv::rectangle(display_for_drawing, roiLente_scaled, cv::Scalar(255, 255, 0), 2);
         cv::circle(display_for_drawing, lens_center_global_scaled, lens_radius_scaled, cv::Scalar(0, 255, 0), 2);

         // Linea teorica (passante per il centro) - in rosso
         cv::Point p1_teorica(
            static_cast<int>(axis_line_global[0] * scale_factor),
            static_cast<int>(axis_line_global[1] * scale_factor)
         );
         cv::Point p2_teorica(
            static_cast<int>(axis_line_global[2] * scale_factor),
            static_cast<int>(axis_line_global[3] * scale_factor)
         );
         cv::line(display_for_drawing, p1_teorica, p2_teorica, cv::Scalar(0, 0, 255), 2);

         // Linea reale della banda - in blu
         cv::Point2f p1_roi(band_analysis.fitted_line[0], band_analysis.fitted_line[1]);
         cv::Point2f p2_roi(band_analysis.fitted_line[2], band_analysis.fitted_line[3]);
         cv::Point2f p1_global = roiToGlobal(p1_roi);
         cv::Point2f p2_global = roiToGlobal(p2_roi);

         cv::Point p1_banda(
            static_cast<int>(p1_global.x * scale_factor),
            static_cast<int>(p1_global.y * scale_factor)
         );
         cv::Point p2_banda(
            static_cast<int>(p2_global.x * scale_factor),
            static_cast<int>(p2_global.y * scale_factor)
         );
//         cv::line(display_for_drawing, p1_banda, p2_banda, cv::Scalar(255, 0, 0), 2);

         // Linea di distanza minima - in verde
         cv::Point2f closest_global = roiToGlobal(band_analysis.closest_point);
         cv::Point closest_scaled(
            static_cast<int>(closest_global.x * scale_factor),
            static_cast<int>(closest_global.y * scale_factor)
         );
         cv::line(display_for_drawing, lens_center_global_scaled, closest_scaled, cv::Scalar(0, 255, 0), 1);

         // Testi informativi (con dimensione font adattata)
         double font_scale = std::max(0.5, 0.7 * scale_factor);
         int font_thickness = std::max(1, static_cast<int>(2 * scale_factor));

         cv::putText(display_for_drawing, "ROI",
            cv::Point(roiLente_scaled.x + 5, roiLente_scaled.y - 5),
            cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 0), font_thickness);

         cv::putText(display_for_drawing, "Linea Teorica (rosso)",
            cv::Point(10, 50),
            cv::FONT_HERSHEY_SIMPLEX, font_scale * 1.2, cv::Scalar(255, 255, 255), font_thickness);

         std::string angle_text = "Angolo: " + std::to_string(normalizzaAngolo(optimal_angle)) + " gradi";
         cv::putText(display_for_drawing, angle_text,
            cv::Point(10, 80),
            cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), font_thickness);

         std::string distance_text = "Distanza dal centro: " +
            std::to_string(static_cast<float>(band_analysis.distance)) + " pixel";
         cv::putText(display_for_drawing, distance_text,
            cv::Point(10, 110),
            cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 255, 0), font_thickness);

         // Aggiungi informazioni sul fattore di scala se l'immagine è stata ridimensionata
         if (scale_factor < 1.0) {
            std::string scale_text = "Zoom: " + std::to_string(static_cast<int>(scale_factor * 100)) + "%";
            cv::putText(display_for_drawing, scale_text,
               cv::Point(10, 140),
               cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 0), font_thickness);
         }

         // Grafica per analyzeBandPosition
         // Cerchietto sul punto più vicino per evidenziarlo
//         cv::circle(display_for_drawing, closest_scaled, 5, cv::Scalar(0, 255, 0), -1);
         cv::circle(display_for_drawing, lens_center_global_scaled, 5, cv::Scalar(0, 255, 0), -1);

         // Aggiungi testo per indicare la posizione della banda
         std::string position_text = "Banda: " +
            std::string(band_analysis.closest_point.y < lens_center.y ? "SOPRA" : "SOTTO") +
            " il centro";
         cv::putText(display_for_drawing, position_text,
            cv::Point(10, 170),
            cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 0), font_thickness);

         // Offset percentuale
         double offset_percent = (band_analysis.distance / lens_radius) * 100.0;
         std::string offset_text = "Offset % rispetto al centro lente: " + cv::format("%.1f%%", offset_percent);
         cv::putText(display_for_drawing, offset_text,
            cv::Point(10, 200),
            cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 0), font_thickness);

         cv::imshow(window_name, display_for_drawing);
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

   // Versione modificata di analyzeBandPosition che usa la stessa metrica
   BandLineResult analyzeBandPosition() {
      std::cout << "\nAnalisi posizione banda chiara usando metrica: "
         << getMetricName(current_metric) << std::endl;

      // Verifica che abbiamo un risultato ottimale salvato
      if (!has_optimal_result) {
         throw std::runtime_error("Nessun risultato ottimale disponibile. Eseguire prima findOptimalOrientation()");
      }

      // USA DIRETTAMENTE I DATI GIÀ CALCOLATI invece di ricalcolare tutto
      const IntegralResult& band_result = optimal_result;

      // Determina la posizione della banda basandosi sulla metrica selezionata
      double band_y_position;
      int band_center_idx;

      switch (current_metric) {
      case EvaluationMetric::INTEGRAL_VALUE:
      case EvaluationMetric::COMBINED_SCORE:
         // Usa il centro della striscia chiara identificata dall'integrale
         if (band_result.bright_stripe_start == -1 || band_result.bright_stripe_end == -1) {
            throw std::runtime_error("Impossibile determinare i limiti della striscia chiara");
         }
         band_center_idx = (band_result.bright_stripe_start + band_result.bright_stripe_end) / 2;
         band_y_position = band_result.positions[band_center_idx];
         break;

      case EvaluationMetric::INTEGRAL_AMPLITUDE:
         // Per l'ampiezza, trova la striscia più ampia
      {
         // Usa direttamente i dati già calcolati in optimal_result
         band_center_idx = (band_result.bright_stripe_start + band_result.bright_stripe_end) / 2;
         band_y_position = band_result.positions[band_center_idx];
      }
      break;

      case EvaluationMetric::MAX_INTEGRAL:
         // Trova il punto di massima intensità
      {
         auto max_it = std::max_element(band_result.profile.begin(), band_result.profile.end());
         band_center_idx = static_cast<int>(std::distance(band_result.profile.begin(), max_it));
         band_y_position = band_result.positions[band_center_idx];
      }
      break;
      }

      // Affina la posizione usando una media pesata intorno al punto identificato
      double weighted_sum = 0;
      double weight_total = 0;
      int window = 10; // finestra di ricerca intorno al punto centrale

      int start_window = std::max(0, band_center_idx - window);
      int end_window = std::min(static_cast<int>(band_result.profile.size()),
         band_center_idx + window + 1);

      for (int i = start_window; i < end_window; i++) {
         double weight = band_result.profile[i];
         weighted_sum += band_result.positions[i] * weight;
         weight_total += weight;
      }

      if (weight_total > 0) {
         band_y_position = weighted_sum / weight_total;
      }

      // Crea il risultato
      BandLineResult result;
      result.slope = 0.0; // La banda è orizzontale dopo la rotazione ottimale
      result.intercept = band_y_position;
      result.distance = std::abs(band_y_position - lens_center.y);
      result.closest_point = cv::Point2f(lens_center.x, static_cast<float>(band_y_position));
      result.r_squared = 1.0; // Non è una regressione, ma un calcolo diretto

      // Crea la linea orizzontale per la visualizzazione (nelle coordinate ruotate)
      float x_start = lens_center.x - lens_radius;
      float x_end = lens_center.x + lens_radius;
      result.fitted_line = cv::Vec4f(x_start, static_cast<float>(band_y_position),
         x_end, static_cast<float>(band_y_position));

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

      // Calcola l'offset dalla posizione teorica centrata
      double theoretical_center_y = lens_center.y;
      double offset_from_center = band_y_position - theoretical_center_y;

      // Output dei risultati
      std::cout << "   Metrica utilizzata: " << getMetricName(current_metric) << std::endl;
      std::cout << "   Valore della metrica: " << getMetricValue(band_result, current_metric) << std::endl;
      std::cout << "   Posizione Y della banda (coordinate ruotate): " << band_y_position << std::endl;
      std::cout << "   Centro teorico Y: " << theoretical_center_y << std::endl;
      std::cout << "   Offset dal centro: " << offset_from_center << " pixel" << std::endl;
      std::cout << "   Distanza assoluta dal centro: " << result.distance << " pixel" << std::endl;
      std::cout << "   Posizione banda: " << (offset_from_center < 0 ? "sopra" : "sotto") << " il centro" << std::endl;

      // Informazioni aggiuntive basate sulla metrica
      std::cout << "   (Dati riutilizzati dal calcolo dell'orientamento ottimale)" << std::endl;

      return result;
   }
};


int main(int argc, char** argv) {
   try {
#ifdef _OPENMP
      // Configura il numero di thread OpenMP (opzionale)
       omp_set_num_threads(6); // Forza 4 thread se necessario
#endif

      std::string nomeFileImg = "imgBiDeg.png";
      //std::string nomeFileImg = "Bi_IRconFiltro.png";
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