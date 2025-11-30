import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import {bundleResourceIO, decodeJpeg} from '@tensorflow/tfjs-react-native';

import {Base64Binary} from '../utils/utils';
import {BITMAP_DIMENSION} from './image-helper';

const modelJson = require('../model/model.json');
const modelWeights = require('../model/weights.bin');

// 0: channel from JPEG-encoded image
// 1: gray scale
// 3: RGB image
const TENSORFLOW_CHANNEL = 3;

export const getModel = async () => {
  try {
    // wait until tensorflow is ready
    await tf.ready();
    // load the trained model
    return await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
  } catch (error) {
    console.log('Could not load model', error);
  }
};

export const convertBase64ToTensor = async (base64) => {
  try {
      const uIntArray = Base64Binary.decode(base64);
      // decode a JPEG-encoded image to a 3D Tensor of dtype
      const decodedImage = decodeJpeg(uIntArray, 3); // [H,W,3]
      // Resize to 224x224
      const resized = tf.image.resizeBilinear(decodedImage, [224, 224]);
      // Reshape to [1,224,224,3]
      const normalized = resized.div(tf.scalar(255));
      return normalized.reshape([1, 224, 224, 3]);
  } catch (error) {
    console.log('Could not convert base64 string to tesor', error);
  }
};

export const startPrediction = async (model, tensor) => {
  try {
    // predict against the model
    const output = await model.predict(tensor);
    console.log('Prediction Output: ', output);
    // return typed array
    return output.dataSync();
  } catch (error) {
    console.log('Error predicting from tesor image', error);
  }
};
