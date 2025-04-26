package net.devk;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;

import java.io.IOException;
import java.nio.file.Paths;

public class Learning {

    public static void main(String[] args) throws IOException {
        ImageFolder dataset = loadDataset("");

        try (Model model = getModel(); Trainer trainer = model.newTrainer(getTrainingConfig())) {

            trainer.setMetrics(new Metrics());
            trainer.initialize(new Shape(1,3,224,224));
//            EasyTrain.fit(trainer,1,dataset,);
        }


    }

    private static ImageFolder loadDataset(String path) throws IOException {
        ImageFolder dataset = ImageFolder.builder().setRepositoryPath(Paths.get(path))
                .addTransform(new Resize(224, 224))
                .addTransform(new ToTensor())
                .setSampling(8, true)
                .build();
        dataset.prepare(new ProgressBar());
        return dataset;
    }

    private static Model getModel() {
        Model model = Model.newInstance("my-demo-mode");
        Block resNet50 = ResNetV1.builder()
                .setImageShape(new Shape(3, 224, 224))
                .setNumLayers(50)
                .setOutSize(2)
                .build();
        model.setBlock(resNet50);
        return model;

    }

    private static TrainingConfig getTrainingConfig() {
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optExecutorService()
                .addTrainingListeners(TrainingListener.Defaults.logging(1));
    }

}
