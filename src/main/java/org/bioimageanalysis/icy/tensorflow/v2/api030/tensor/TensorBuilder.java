package org.bioimageanalysis.icy.tensorflow.v2.api030.tensor;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.ByteDataBuffer;
import org.tensorflow.ndarray.buffer.DoubleDataBuffer;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.ndarray.buffer.IntDataBuffer;
import org.tensorflow.ndarray.buffer.LongDataBuffer;
import org.tensorflow.ndarray.impl.buffer.raw.RawDataBufferFactory;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;


/**
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando 
 */
public final class TensorBuilder
{
    /**
     * Utility class.
     */
    private TensorBuilder()
    {
    }

    /**
     * Creates {@link TType} instance with the same size and information as the given {@link INDArray}.
     * 
     * @param sequence
     *        The sequence which the created tensor is filled with.
     * @return The created tensor.
     * @throws IllegalArgumentException
     *         If the type of the sequence is not supported.
     */
    public static TType build(org.bioimageanalysis.icy.deeplearning.tensor.Tensor tensor) throws IllegalArgumentException
    {
    	return build(tensor.getDataAsNDArray());
    }

    /**
     * Creates {@link TType} instance with the same size and information as the given {@link INDArray}.
     * 
     * @param sequence
     *        The sequence which the created tensor is filled with.
     * @return The created tensor.
     * @throws IllegalArgumentException
     *         If the type of the sequence is not supported.
     */
    public static TType build(INDArray array) throws IllegalArgumentException
    {
        // Create an Icy sequence of the same type of the tensor
    	if (array.dataType() == DataType.INT8 || array.dataType() == DataType.UINT8) {
            return buildByte(array);
    	} else if (array.dataType() == DataType.INT32) {
            return buildInt(array);
    	} else if (array.dataType() == DataType.FLOAT) {
            return buildFloat(array);
    	} else if (array.dataType() == DataType.DOUBLE) {
            return buildDouble(array);
    	} else if (array.dataType() == DataType.BOOL) {
            return buildBoolean(array);
    	} else if (array.dataType() == DataType.INT64) {
            return buildLong(array);
    	} else {
            throw new IllegalArgumentException("Unsupported tensor type: " + array.dataType().toString());
    	}
    }

    /**
     * Creates a {@link TType} tensor of type {@link TUint8} from an {@link INDArray} of type {@link DataType#BYTE} or {@link DataType#UBYTE}.
     * 
     * @param array
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    public static TUint8 buildByte(INDArray array) throws IllegalArgumentException
    {
        if (array.dataType() != DataType.INT8 && array.dataType() != DataType.UINT8)
            throw new IllegalArgumentException("Tensor is not of byte type: " + array.dataType().toString());

        ByteDataBuffer dataBuffer = RawDataBufferFactory.create(array.data().asBytes(), false);
        TUint8 tensor = Tensor.of(TUint8.class, Shape.of(array.shape()), dataBuffer);
		return tensor;
    }

    /**
     * Creates a {@link TType} tensor of type {@link TInt32} from an {@link INDArray} of type 
     * {@link DataType#INT} or
     * {@link DataType#UINT}.
     * 
     * @param array
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    public static TInt32 buildInt(INDArray array) throws IllegalArgumentException
    {
        if (array.dataType() != DataType.INT32 && array.dataType() != DataType.UINT32)
            throw new IllegalArgumentException("Image is not of int type: " + array.dataType());

        IntDataBuffer dataBuffer = RawDataBufferFactory.create(array.data().asInt(), false);
        TInt32 tensor = TInt32.tensorOf(Shape.of(array.shape()), dataBuffer);
		return tensor;
    }

    /**
     * Creates a {@link Tensor} of type {@link TInt64} from an {@link INDArray} of type {@link DataType#LONG}
     * 
     * @param sequence
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    private static TInt64 buildLong(INDArray array) throws IllegalArgumentException
    {
        if (array.dataType() != DataType.INT32 && array.dataType() != DataType.UINT32)
            throw new IllegalArgumentException("Image is not of int type: " + array.dataType());

        LongDataBuffer dataBuffer = RawDataBufferFactory.create(array.data().asLong(), false);
        TInt64 tensor = TInt64.tensorOf(Shape.of(array.shape()), dataBuffer);
		return tensor;
    }

    /**
     * Creates a {@link TType} tensor of type {@link TFloat32} from an {@link INDArray} of type {@link DataType#FLOAT}.
     * 
     * @param sequence
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    public static TFloat32 buildFloat(INDArray array) throws IllegalArgumentException
    {
        if (array.dataType() != DataType.FLOAT)
            throw new IllegalArgumentException("Tensor is not of float type: " + array.dataType());

        FloatDataBuffer dataBuffer = RawDataBufferFactory.create(array.data().asFloat(), false);
        TFloat32 tensor = TFloat32.tensorOf(Shape.of(array.shape()), dataBuffer);
		return tensor;
    }

    /**
     * Creates a {@link Tensor} of type {@link TFloat64} from an {@link INDArray} of type {@link DataType#DOUBLE}.
     * 
     * @param array
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    private static TFloat64 buildDouble(INDArray array) throws IllegalArgumentException
    {
        if (array.dataType() != DataType.DOUBLE)
            throw new IllegalArgumentException("Tensor is not of float type: " + array.dataType());

        DoubleDataBuffer dataBuffer = RawDataBufferFactory.create(array.data().asDouble(), false);
        TFloat64 tensor = TFloat64.tensorOf(Shape.of(array.shape()), dataBuffer);
		return tensor;
    }

    /**
     * Creates a {@link Tensor} of type {@link TBool} from an {@link INDArray}.
     * 
     * @param array
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    private static TBool buildBoolean(INDArray array) throws IllegalArgumentException
    {
        if (array.dataType() != DataType.BOOL)
            throw new IllegalArgumentException("Tensor is not of boolean type: " + array.dataType());

        ByteDataBuffer dataBuffer = RawDataBufferFactory.create(array.data().asBytes(), false);
        TBool tensor = Tensor.of(TBool.class, Shape.of(array.shape()), dataBuffer);
		return tensor;
    }
}
