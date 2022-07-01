import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';

console.log(tf.version.tfjs);
console.log(mobilenet.version);

mobilenet.load().then((model) => {
  const img = document.getElementById('truck') as HTMLImageElement;
  model.classify(img).then((result) => {
    console.log(result);
    const isTruck = result.some((r) => r.className.includes('truck'));
    if (isTruck) {
      console.log('truck');
    }
  });
});
