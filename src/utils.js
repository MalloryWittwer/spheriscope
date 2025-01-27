import { dot, norm, cross, multiply, inv } from "mathjs";

const matrixMul = (m1, m2) => {
  return multiply(m1, m2);
};

const matrixInv = (m) => {
  return inv(m);
};

const spher2cart = (tp) => {
  const [t, p] = tp;
  const x = Math.cos(p) * Math.sin(t);
  const y = Math.sin(p) * Math.sin(t);
  const z = Math.cos(t);
  return [x, y, z];
};

const cart2spher = (xyz) => {
  const [x, y, z] = xyz;
  const t = Math.acos(z);
  const p = Math.atan2(y, x);
  return [t, p];
};

const stereoProjection = (xyz) => {
  let [x, y, z] = xyz;
  if (z === 0) {
    return [0, 0];
  }
  const X = x / (1 - z);
  const Y = y / (1 - z);
  return [X, Y];
};

const invStereoProjection = (XY) => {
  const [X, Y] = XY;
  const r = 1 + X ** 2 + Y ** 2;
  const x = (2 * X) / r;
  const y = (2 * Y) / r;
  const z = (-1 + X ** 2 + Y ** 2) / r;
  return [x, y, z];
};

const orthoProjection = (tp) => {
  let [t, p] = tp;
  const t0 = 0;
  const tshifted = t + Math.PI / 2;
  // const tshifted = t;

  // const cosC =
  //   Math.sin(t0) * Math.sin(tshifted) +
  //   Math.cos(t0) * Math.cos(tshifted) * Math.cos(p);
  // if (cosC > 0) {
  //   return [10000, 10000]; // Outside of view
  // }
  // const cosC =
  //   Math.sin(t0) * Math.sin(tshifted) +
  //   Math.cos(t0) * Math.cos(tshifted) * Math.cos(p);
  // if (cosC < 0) {
  //   return [10000, 10000]; // Outside of view
  // }

  const X = Math.cos(tshifted) * Math.sin(p);
  const Y =
    Math.cos(t0) * Math.sin(tshifted) -
    Math.sin(t0) * Math.cos(tshifted) * Math.cos(p);
  return [X, Y];
};

const invOrthoProjection = (XY) => {
  let [X, Y] = XY;
  let rho;
  rho = Math.sqrt(X ** 2 + Y ** 2);
  const angle = Math.atan2(Y, X);
  if (rho >= 0.5) {
    rho = 0.5;
    X = rho * Math.cos(angle);
    Y = rho * Math.sin(angle);
  }
  const t0 = 0;
  const c = Math.asin(rho);
  const t =
    Math.asin(
      Math.cos(c) * Math.sin(t0) + (Y * Math.sin(c) * Math.cos(t0)) / rho,
    ) -
    Math.PI / 2;
  const p = Math.atan(
    ((X * Math.sin(c)) / rho) * Math.cos(c) * Math.cos(t0) -
    Y * Math.sin(c) * Math.sin(t0),
  );
  return [t, p];
};

const equalAreaProjection = (tp) => {
  let [t, p] = tp;
  const Y = Math.sin(t - Math.PI / 2);
  return [p, Y];
};

const invEqualAreaProjection = (XY) => {
  let [X, Y] = XY;
  X = clamp(X, -Math.PI, Math.PI);
  Y = clamp(Y, -0.99, 0.99);
  const t = Math.asin(-Y) - Math.PI / 2;
  const p = X - Math.PI;
  return [t, p];
};

const arraysEqual = (a, b) => {
  for (let i = 0; i < a.length; ++i) {
    if (a[i] !== b[i]) return false;
  }
  return true;
};

const crossVectorNormed = (v1, v2) => {
  const cv = cross(v1, v2);
  const cvNorm = norm(cv);
  const cvNormed = cv.map((x) => x / cvNorm);
  return cvNormed;
};

const angleBetweenVectors = (v1, v2) => {
  return Math.acos(dot(v1, v2) / norm(v1) / norm(v2));
};

const matrixRot = (matrix, vect) => {
  const rotatedVect = [
    matrix[0][0] * vect[0] + matrix[0][1] * vect[1] + matrix[0][2] * vect[2],
    matrix[1][0] * vect[0] + matrix[1][1] * vect[1] + matrix[1][2] * vect[2],
    matrix[2][0] * vect[0] + matrix[2][1] * vect[1] + matrix[2][2] * vect[2],
  ];
  return rotatedVect;
};

const clamp = (num, min, max) => Math.min(Math.max(num, min), max);

export {
  spher2cart,
  cart2spher,
  stereoProjection,
  invStereoProjection,
  orthoProjection,
  invOrthoProjection,
  equalAreaProjection,
  invEqualAreaProjection,
  matrixRot,
  matrixMul,
  matrixInv,
  angleBetweenVectors,
  crossVectorNormed,
  arraysEqual,
};
