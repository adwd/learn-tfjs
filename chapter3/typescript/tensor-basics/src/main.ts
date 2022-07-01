import * as tf from '@tensorflow/tfjs';
import './style.css';
import './copied';

const app = document.querySelector<HTMLDivElement>('#app')!;

app.innerHTML = `
  <h1>Hello Vite!</h1>
  <a href="https://vitejs.dev/guide/features.html" target="_blank">Documentation</a>
`;

// tensor
const dataArray = [8, 6, 7, 5, 3, 0, 9];
const first = tf.tensor(dataArray);
const first_again = tf.tensor1d(dataArray);

const float32 = tf.tensor([1.1, 2.0], undefined, 'float32');
const int32 = tf.tensor([1, 2], undefined, 'int32');
const bool = tf.tensor([true, false], undefined, 'bool');
const is_bool = tf.tensor([true, false], undefined, 'int32');
console.log(await is_bool.data()); // [1, 0]

const second = tf.tensor1d([1, 2, 3, 4]);
// {rank: 1, size: 4, dtype: 'float32'}
console.log({
  rank: second.rank,
  size: second.size,
  dtype: second.dtype,
});
try {
  const nope = tf.tensor1d([[1, 2]]);
} catch (e) {
  // Error: tensor1d() requires values to be a flat/TypedArray
  //  at Module.tensor1d (tensor1d.ts:45:11)
  //  at main.ts:25:19
  console.log(e);
}

const a = tf.tensor([1, 0, 0, 0, -1, 0, 1, 0, 0]);
const b = tf.tensor([
  [1, 0, 0],
  [0, -1, 0],
  [1, 0, 0],
]);

// 2次元テンソルに変換される
const c = tf.tensor([1, 0, 0, 0, -1, 0, 1, 0, 0], [3, 3]);
const d = tf.tensor([1, 0, 0, 0, -1, 0, 1, 0, 0], [3, 3], 'int32');
const e = d.asType('float32');

const tensorArray = [];
for (let i = 0; i < 10; i++) {
  tensorArray.push(tf.tensor([i, i, i]));
}

tf.dispose(tensorArray);

const t = tf.tensor([
  [1, 2],
  [3, 4],
  [5, 6],
]);

const x1 = await t.array();
const x2 = t.arraySync();

const x3 = await t.data();
const x4 = t.dataSync();

console.log(t, x1, x3);
t.print();

const mat1 = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9],
];

const mat2 = [
  [1, 3, 5],
  [2, 4, 6],
  [3, 5, 7],
];

tf.matMul(mat1, mat2).print();
