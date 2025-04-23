package net.devk;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

/**
 * Hello world!
 */
public class MnistInference {
    public static void main(String[] args) throws IOException, MalformedModelException, TranslateException {
        var img = ImageFactory.getInstance().fromFile(Paths.get("/home/ali/mnist/2.png"));
        img.getWrappedImage();

        Path modelDir = Paths.get("build/mlp");
        Model model = Model.newInstance("mlp", Device.gpu());
        model.setBlock(new Mlp(28 * 28, 10, new int[] { 128, 64 }));
        model.load(modelDir);

        Translator<Image, Classifications> translator = new Translator<Image, Classifications>() {

            @Override
            public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
                // Convert Image to NDArray
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
                return new NDList(NDImageUtils.toTensor(array));
            }

            @Override
            public Classifications processOutput(TranslatorContext ctx, NDList list) throws Exception {
                // Create a Classifications with the output probabilities
                NDArray probabilities = list.singletonOrThrow().softmax(0);
                List<String> classNames = IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
                return new Classifications(classNames, probabilities);
            }

            @Override
            public Batchifier getBatchifier() {
                // The Batchifier describes how to combine a batch together
                // Stacking, the most common batchifier, takes N [X1, X2, ...] arrays to a
                // single [N, X1, X2, ...] array
                return Batchifier.STACK;
            }

        };

        var predictor = model.newPredictor(translator);

        var classifications = predictor.predict(img);

        List<String> classNames = classifications.getClassNames();
        for (int i = 0; i < classNames.size(); i++) {
            System.out.printf("%s : %.20f%n", classNames.get(i), classifications.getProbabilities().get(i));
        }

    }
}
