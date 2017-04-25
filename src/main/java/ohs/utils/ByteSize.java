package ohs.utils;

import java.text.DecimalFormat;
import java.util.BitSet;

public class ByteSize {

	public static enum Type {
		KILO, MEGA, GIGA, TERA, PETA
	}

	public static final int DENOMINATOR = 1024;

	public static void main(String[] args) {
		System.out.println(new ByteSize(Byte.BYTES * 1l * Integer.MAX_VALUE));
		System.out.println(new ByteSize(Short.BYTES * 1l * Integer.MAX_VALUE));
		System.out.println(new ByteSize(Double.BYTES * 1l * Integer.MAX_VALUE));
		System.out.println(new ByteSize(Integer.BYTES * 1l * Integer.MAX_VALUE));
		System.out.println(new ByteSize(Long.BYTES * 1l * Integer.MAX_VALUE));

		System.out.println(new ByteSize(10, Type.MEGA).toString());
		System.out.println(new ByteSize(10, Type.MEGA).getBytes());
		System.out.println(new ByteSize(10485760).toString());
		
		
		int size = 100;
		
		BitSet bs = new BitSet();
		
		for(int i = 0 ;i < size;i++){
			bs.set(i);
		}
		
		System.out.println(new ByteSize(bs.size() / 8));
		System.out.println(new ByteSize(Integer.BYTES  * size));
		
	}

	private long bytes = 0;

	private double[] sizes = new double[Type.values().length];

	public ByteSize(long bytes) {
		this.bytes = bytes;
		compute(bytes);
	}

	public ByteSize(long size, Type type) {
		long amount = DENOMINATOR;
		int loc = type.ordinal();
		for (int i = 1; i <= loc; i++) {
			amount *= DENOMINATOR;
		}

		bytes = (long) (size * amount);

		compute(bytes);
	}

	private void compute(long bytes) {
		for (int i = 0; i < sizes.length; i++) {
			if (i == 0) {
				sizes[i] = 1f * bytes / DENOMINATOR;
			} else {
				sizes[i] = 1f * sizes[i - 1] / DENOMINATOR;
			}
		}
	}

	public long getBytes() {
		return bytes;
	}

	public double getSize(Type type) {
		return sizes[type.ordinal()];
	}

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < Type.values().length; i++) {
			int j = Type.values().length - i - 1;
			Type type = Type.values()[j];
			if (sizes[j] >= 1) {
				DecimalFormat df = new DecimalFormat(".00");
				sb.append(String.format("[%f %cBs]", sizes[j], type.toString().charAt(0)));
				break;
			}
		}

		if (sb.length() == 0) {
			sb.append(String.format("[%d] BTs", bytes));
		}

		return sb.toString();
	}
}
