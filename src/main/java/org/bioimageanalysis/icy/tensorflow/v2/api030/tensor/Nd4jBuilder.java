package org.bioimageanalysis.icy.tensorflow.v2.api030.tensor;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;


/**
 * @author Carlos GArcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public final class Nd4jBuilder
{
    /**
     * Utility class.
     */
    private Nd4jBuilder()
    {
    }

    public static INDArray build(TType tensor) throws IllegalArgumentException
    {
    	if (tensor instanceof TUint8)
        {
            return buildFromTensorByte((TUint8) tensor);
        }
        else if (tensor instanceof TInt32)
        {
            return buildFromTensorInt((TInt32) tensor);
        }
        else if (tensor instanceof TFloat32)
        {
            return buildFromTensorFloat((TFloat32) tensor);
        }
        else if (tensor instanceof TFloat64)
        {
            return buildFromTensorDouble((TFloat64) tensor);
        }
        else if (tensor instanceof TBool)
        {
            return buildFromTensorBoolean((TBool) tensor);
        }
        else if (tensor instanceof TInt64)
        {
            return buildFromTensorLong((TInt64) tensor);
        }
        else
        {
            throw new IllegalArgumentException("Unsupported tensor type: " + tensor.dataType().name());
        }
    }

    private static INDArray buildFromTensorBoolean(TBool tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		byte[] flatImageArray = new byte[(int) size];
		// Copy data from tensor to array
        tensor.asRawTensor().data().read(flatImageArray);
		return Nd4j.create(flatImageArray, tensorShape, DataType.BOOL);
    }

    private static INDArray buildFromTensorByte(TUint8 tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		byte[] flatImageArray = new byte[(int) size];
		// Copy data from tensor to array
        tensor.asRawTensor().data().read(flatImageArray);
		return Nd4j.create(flatImageArray, tensorShape, DataType.INT8);
    }

    private static INDArray buildFromTensorInt(TInt32 tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		int[] flatImageArray = new int[(int) size];
		// Copy data from tensor to array
        tensor.asRawTensor().data().asInts().read(flatImageArray);
		return Nd4j.create(flatImageArray, tensorShape, DataType.INT32);
    }

    private static INDArray buildFromTensorFloat(TFloat32 tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		float[] flatImageArray = new float[(int) size];
		// Copy data from tensor to array
        tensor.asRawTensor().data().asFloats().read(flatImageArray);
		return Nd4j.create(flatImageArray, tensorShape, DataType.FLOAT);
    }

    private static INDArray buildFromTensorDouble(TFloat64 tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		double[] flatImageArray = new double[(int) size];
		// Copy data from tensor to array
        tensor.asRawTensor().data().asDoubles().read(flatImageArray);
		return Nd4j.create(flatImageArray, tensorShape, DataType.DOUBLE);
    }

    private static INDArray buildFromTensorLong(TInt64 tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		long[] flatImageArray = new long[(int) size];
		// Copy data from tensor to array
        tensor.asRawTensor().data().asLongs().read(flatImageArray);
		return Nd4j.create(flatImageArray, tensorShape, DataType.INT64);
    }
}
