import * as tf from '@tensorflow/tfjs';
import { INCEPTION_CLASSES } from './labels';

import './style.css';

const app = document.querySelector<HTMLDivElement>('#app')!;

app.innerHTML = `
  <h1>Hello Vite!</h1>
  <a href="https://vitejs.dev/guide/features.html" target="_blank">Documentation</a>
`;

await tf.ready();
tf.tidy(() => {
  tf.loadLayersModel('/model/ttt_model.json').then((model) => {
    // 3つの盤面の状態
    const emptyBoard = tf.zeros([9]);
    const betterBlockMe = tf.tensor([-1, 0, 0, 1, 1, -1, 0, 0, -1]);
    const goForTheKill = tf.tensor([1, 0, 1, 0, -1, -1, -1, 0, 1]);

    // 状態を[3, 9]の形状にスタック
    const matches = tf.stack([emptyBoard, betterBlockMe, goForTheKill]);
    const result = model.predict(matches);

    if (Array.isArray(result)) {
      result.forEach((r) => {
        r.reshape([3, 3, 3]).print();
      });
    } else {
      result.reshape([3, 3, 3]).print();
    }
  });
});

console.log(tf.memory().numTensors);

// inception v3
const modelPath = 'https://tfhub.dev/google/tfjs-model/imagenet/inception_v3/classification/3/default/1';
tf.tidy(() => {
  tf.loadGraphModel(modelPath, { fromTFHub: true }).then((model) => {
    const imageAndResultElements = [
      ['mystery', 'mystery-answer'],
      ['mystery2', 'mystery2-answer'],
      ['mystery3', 'mystery3-answer'],
    ];

    imageAndResultElements.forEach(([imageEl, answerEl]) => {
      const mysteryImage = document.getElementById(imageEl) as HTMLImageElement;
      const myTensor = tf.browser.fromPixels(mysteryImage);
      // Inception v3 expects an image resized to 299x299
      const readyfied = tf.image.resizeBilinear(myTensor, [299, 299], true).div(255).reshape([1, 299, 299, 3]);

      const result = model.predict(readyfied) as tf.Tensor<tf.Rank>;
      result.print(); // useless

      const { values, indices } = tf.topk(result, 3);
      indices.print();

      // Let's hear those winners
      const winners = indices.dataSync();
      const answer = document.getElementById(answerEl);
      answer!.innerText = `
              🥇 First place ${INCEPTION_CLASSES[winners[0]]},
              🥈 Second place ${INCEPTION_CLASSES[winners[1]]},
              🥉 Third place ${INCEPTION_CLASSES[winners[2]]}
            `;
    });
  });
});

console.log(tf.memory().numTensors);

// object localization
tf.tidy(() => {
  tf.loadLayersModel('/model/tfjs_quant_uint8/model.json').then((model) => {
    const petImage = document.getElementById('pet') as HTMLImageElement;
    const myTensor = tf.browser.fromPixels(petImage);
    // Model expects 256x256 0-1 value 3D tensor
    const readyfied = tf.image.resizeNearestNeighbor(myTensor, [256, 256], true).div(255).reshape([1, 256, 256, 3]);

    const result = model.predict(readyfied);
    // Model returns top left and bottom right
    console.log('object location:');
    result.print();

    // Draw box on canvas
    const detection = document.getElementById('detection') as HTMLCanvasElement;
    const imgWidth = petImage.width;
    const imgHeight = petImage.height;
    detection.width = imgWidth;
    detection.height = imgHeight;
    const box = result.dataSync();
    const startX = box[0] * imgWidth;
    const startY = box[1] * imgHeight;
    const width = (box[2] - box[0]) * imgWidth;
    const height = (box[3] - box[1]) * imgHeight;
    const ctx = detection.getContext('2d')!;
    ctx.strokeStyle = '#0F0';
    ctx.lineWidth = 4;
    ctx.strokeRect(startX, startY, width, height);

    // challenge
    {
      const tHeight = myTensor.shape[0];
      const tWidth = myTensor.shape[1];

      console.log({ box });
      const tStartX = box[0] * tWidth;
      const tStartY = box[1] * tHeight;
      const cropWidth = parseInt((box[2] - box[0]) * tWidth, 0);
      const cropHeight = parseInt((box[3] - box[1]) * tHeight, 0);

      const startPos = [tStartY, tStartX, 0];
      const cropSize = [cropHeight, cropWidth, 3];

      const cropped = tf.slice(myTensor, startPos, cropSize);

      const readyFace = tf.image.resizeBilinear(cropped, [96, 96], true).div(255);
      const out = document.getElementById('out') as HTMLCanvasElement;
      tf.browser.toPixels(readyFace, out);
    }
  });
});
