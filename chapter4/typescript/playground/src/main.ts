import * as tf from '@tensorflow/tfjs';
import './style.css';

const app = document.querySelector<HTMLDivElement>('#app')!;

app.innerHTML = `
  <h1>Hello Vite!</h1>
  <a href="https://vitejs.dev/guide/features.html" target="_blank">Documentation</a>
`;

const bigMess = tf.randomUniform<tf.Rank.R3>([400, 400, 3], 0, 255, 'int32');
const myCanvas = document.getElementById('randomness') as HTMLCanvasElement;
tf.browser.toPixels(bigMess, myCanvas).then(() => {
  bigMess.dispose();
  console.log('Make sure we cleaned up', tf.memory().numTensors);
});

tf.tidy(() => {
  const lil = tf.tensor([
    [[1], [0]],
    [[0], [1]],
  ]);

  const big = lil.tile([100, 100, 1]);
});

tf.tidy(() => {
  // Simply read from the DOM
  const gantImage = document.getElementById('gant') as HTMLImageElement;
  const gantTensor = tf.browser.fromPixels(gantImage);
  console.log(`Successful conversion from DOM to a ${gantTensor.shape} tensor`);

  // Now load an image object in JavaScript
  const cake = new Image();
  cake.crossOrigin = 'anonymous';
  cake.src = '/cake.jpg';
  cake.onload = () => {
    const cakeTensor = tf.browser.fromPixels(cake);
    console.log(`Successful conversion from Image() to a ${cakeTensor.shape} tensor`);
  };
});

tf.tidy(() => {
  // Simple Tensor Flip
  const lemonadeImage = document.getElementById('lemonade');
  const lemonadeCanvas = document.getElementById('lemonadeCanvas');
  const lemonadeTensor = tf.browser.fromPixels(lemonadeImage);
  const flippedLemonadeTensor = tf.reverse(lemonadeTensor, 1);
  tf.browser.toPixels(flippedLemonadeTensor, lemonadeCanvas).then(() => {
    lemonadeTensor.dispose();
    flippedLemonadeTensor.dispose();
  });
});

tf.tidy(() => {
  // Batch Tensor Flip
  const cakeImage = document.getElementById('cake');
  const cakeCanvas = document.getElementById('cakeCanvas');
  const flipCake = tf.tidy(() => {
    const cakeTensor = tf.expandDims(tf.browser.fromPixels(cakeImage).asType('float32'));
    return tf.squeeze(tf.image.flipLeftRight(cakeTensor)).asType('int32');
  });
  tf.browser.toPixels(flipCake, cakeCanvas).then(() => {
    flipCake.dispose();
  });
});

// Simple Tensor Flip
const newSize = [768, 560]; // 4x larger
const littleGantImage = document.getElementById('littleGant');
const nnCanvas = document.getElementById('nnCanvas');
const blCanvas = document.getElementById('blCanvas');
const gantTensor = tf.browser.fromPixels(littleGantImage);

const nnResizeTensor = tf.image.resizeNearestNeighbor(gantTensor, newSize, true);
tf.browser.toPixels(nnResizeTensor, nnCanvas).then(() => {
  nnResizeTensor.dispose();
});

const blResizeTensor = tf.image.resizeBilinear(gantTensor, newSize, true);
const blResizeTensorInt = blResizeTensor.asType('int32');
tf.browser.toPixels(blResizeTensorInt, blCanvas).then(() => {
  blResizeTensor.dispose();
  blResizeTensorInt.dispose();
});

// All done with ya
gantTensor.dispose();

// challenge
const ch = tf.randomUniform<tf.Rank.R2>([400, 400], 0, 255, 'int32');
const { values: ch2 } = tf.topk(ch, 400);
const chCanvas = document.getElementById('challenge') as HTMLCanvasElement;
tf.browser.toPixels(ch2, chCanvas).then(() => {
  ch.dispose();
  ch2.dispose();
  console.log('Make sure we cleaned up', tf.memory().numTensors);
});
