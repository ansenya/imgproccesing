import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;

public class Main {
    // Загружаем библиотеку OpenCV, а так же проеверяем версию библиотеки.
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("OpenCV version: " + Core.VERSION);
        clearFolder();
    }

    public static void main(String[] args) throws InterruptedException {
        // Создаем окно для просмотра изображения.
        JFrame window = new JFrame("Window:");
        // Создаем контейнер для изображения.
        JLabel screen = new JLabel();
        // Устанавливаем операцию закрытия по умолчанию.
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        // Делаем окно отображения контента видимым.
        window.setVisible(true);

        // Инициализируем видеопоток.
        VideoCapture cap = new VideoCapture(0);

        // Инициализируем переменные.
        Mat frame = new Mat();
        Mat frameResized = new Mat();
        MatOfByte buf = new MatOfByte();
        float minProbability = 0.5f;
        float threshold = 0.3f;
        int height;
        int width;
        ImageIcon ic;
        BufferedImage img;
        MatOfInt indices = new MatOfInt();
        int k = 0;

        // Загружаем файл с наименованиями классов.
        String path = "src/yolov4/yolov4.names";
        List<String> labels = labels(path);
        int amountOfClasses = labels.size();

        // Генерируем цвет для каждого класса.
        Scalar[] colors = generateColors(amountOfClasses);

        /// Инициализируем сверточную нейронную сеть.
        String cfgPath = "src/yolov4/yolov4.cfg";
        String weightsPath = "src/yolov4/yolov4.weights";
        Net network = Dnn.readNetFromDarknet(cfgPath, weightsPath);
        network.setPreferableBackend(Dnn.DNN_BACKEND_CUDA);
        network.setPreferableTarget(Dnn.DNN_TARGET_CUDA);


        // Извлекаем наименования выходных слоев.
        List<String> outputLayersNames = getOutputLayerNames(network);

        // Извлекаем индексы выходных слоев.
        MatOfInt outputLayersIndexes = network.getUnconnectedOutLayers();
        int amountOfOutputLayers = outputLayersIndexes.toArray().length;

        // В бесконечном цикле обрабатываем поступающие кадры из видеопотока.
        while (true) {
            // Извлекаем кадр из видеопотока.
            cap.read(frame);
            height = frame.height();
            width = frame.width();

            // Изменяем размер кадра для уменьшения нагрузки на нейронную сеть.
            Imgproc.resize(frame, frameResized, new Size(256, 256));

            // Подаём blob на вход нейронной сети.
            network.setInput(Dnn.blobFromImage(frameResized, 1 / 255.0));

            // Извлекаем данные с выходных слоев нейронной сети.
            List<Mat> outputFromNetwork = new ArrayList<>();
            network.forward(outputFromNetwork, outputLayersNames);


            // Обнаруживаем объекты на изображении.
            List<Integer> classIndexes = new ArrayList<>();

            List<Float> confidencesList = new ArrayList<>();
            MatOfFloat confidences = new MatOfFloat();

            List<Rect2d> boundingBoxesList = new ArrayList<>();
            MatOfRect2d boundingBoxes = new MatOfRect2d();

            // Проходим через все предсказания из выходных слоёв по очереди.
            // В цикле проходим через слои:
            for (Mat output : outputFromNetwork) {
                // Проходимся по всем предсказаниям.
                for (int i = 0; i < output.rows(); i++) {
                    Mat scores = output.row(i).colRange(5, output.cols());
                    Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                    Point classPoint = mm.maxLoc;
                    double confidence = mm.maxVal;

                    // Фильтруем предсказания по порогу уверенности.
                    if (confidence > minProbability) {
                        int centerX = (int) (output.row(i).get(0, 0)[0] * width);
                        int centerY = (int) (output.row(i).get(0, 1)[0] * height);
                        int boxWidth = (int) (output.row(i).get(0, 2)[0] * width);
                        int boxHeight = (int) (output.row(i).get(0, 3)[0] * height);
                        int left = centerX - boxWidth / 2;
                        int top = centerY - boxHeight / 2;

                        classIndexes.add((int) classPoint.x);
                        confidencesList.add((float) confidence);
                        boundingBoxesList.add(new Rect2d(left, top, boxWidth, boxHeight));
                    }
                }
            }

            // Применяем алгоритм подавления немаксимумов.
            boundingBoxes.fromList(boundingBoxesList);
            confidences.fromList(confidencesList);
            Dnn.NMSBoxes(boundingBoxes, confidences, minProbability, threshold, indices);


            // Если алгоритм выявил ограничительные рамки,
            if (indices.size().height > 0) {
                List<Integer> indicesList = indices.toList();
                List<Rect2d> boundingList = boundingBoxes.toList();
                // то наносим выявленные рамки на изображения.
                for (int i = 0; i < indicesList.size(); i++) {
                    // Ограничительная рамка
                    Rect rect = new Rect(
                            (int) boundingList.get(indicesList.get(i)).x,
                            (int) boundingList.get(indicesList.get(i)).y,
                            (int) boundingList.get(indicesList.get(i)).width,
                            (int) boundingList.get(indicesList.get(i)).height
                    );

                    // Извлекаем индекс выявленгого класса (объекта на изображении).
                    int classIndex = classIndexes.get(indices.toList().get(i));

                    // Наносим ограничительную рамку.
                    Imgproc.rectangle(frame, rect, colors[classIndex], 2);

                    // Форматируем строку для нанесения на изображение:
                    // Выявленный клас: вероятность
                    String label = labels.get(classIndex) + ": " + String.format("%.2f", confidences.toList().get(i));
                    // Инициализируем точку для нанесения текста.
                    Point textPoint = new Point(
                            (int) boundingList.get(indices.toList().get(i)).x,
                            (int) boundingList.get(indices.toList().get(i)).y - 10
                    );
                    // Наносим текст на изображение.
                    Imgproc.putText(frame, label, textPoint, 1, 1.5, colors[classIndex]);
//                    System.out.println(label);
                }
            }

            // Преобразуем Mat в BufferedImage для отображения в окне.
            Imgcodecs.imencode(".png", frame, buf);
            if (k%60==0){
                Imgproc.resize(frame, frame, new Size(frame.width()*0.5, frame.height()*0.5));
                Imgcodecs.imwrite("src/out/" + k + ".png", frame);
            }
            k++;
            ic = new ImageIcon(buf.toArray());

            // Отображаем изображение в окне.
            screen.setIcon(ic);
            screen.repaint();
            window.setContentPane(screen);
            window.pack();
        }
    }

    // Функция для парсинга файла coco.names.
    public static List<String> labels(String path) {
        List<String> labels = new ArrayList();
        try {
            Scanner scnLabels = new Scanner(new File(path));
            while (scnLabels.hasNext()) {
                String label = scnLabels.nextLine();
                labels.add(label);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return labels;
    }

    // Функция для генерации цветов
    public static Scalar[] generateColors(int amountOfClasses) {
        Scalar[] colors = new Scalar[amountOfClasses];
        Random random = new Random();
        for (int i = 0; i < amountOfClasses; i++) {
            int r = random.nextInt(256);
            int g = random.nextInt(256);
            int b = random.nextInt(256);
            colors[i] = new Scalar(r, g, b);
        }
        return colors;
    }

    // Метод для извлечения наименований выходных слоев.
    public static List<String> getOutputLayerNames(Net network) {
        List<String> layersNames = network.getLayerNames();
        List<String> outputLayersNames = new ArrayList<>();
        List<Integer> unconnectedLayersIndexes = network.getUnconnectedOutLayers().toList();
        for (int i : unconnectedLayersIndexes) {
            outputLayersNames.add(layersNames.get(i - 1));
        }
        return outputLayersNames;
    }

    // Очищает папку "out"
    public static void clearFolder(){
        String folderPath = "/src/out";
        Path path = Paths.get(folderPath);
        try {
            Files.delete(path);
            System.out.println("Папка успешно удалена.");
        } catch (Exception e) {
            System.out.println("Ошибка при удалении папки: " + e.getMessage());
        }
    }
}