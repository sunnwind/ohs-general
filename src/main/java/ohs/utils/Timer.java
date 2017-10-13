package ohs.utils;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * Simple class for measuring elapsed ms.
 */
public class Timer {

	public static Timer newTimer() {
		Timer ret = new Timer();
		ret.start();
		return ret;
	}

	private int n = 0;

	private long start = 0, end = 0, ms = 0;

	public Timer() {

	}

	public Timer(long time) {
		start = 0;
		end = time;
		this.ms = time;
	}

	public Timer(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public Timer accumStop() { // Stop and accumulate ms
		end = System.currentTimeMillis();
		ms += end - start;
		n++;
		return this;
	}

	public long getEnd() {
		return end;
	}

	public long getMs() {
		return ms;
	}

	public int getN() {
		return n;
	}

	public long getStart() {
		return start;
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		start = ois.readLong();
		end = ois.readLong();
		ms = ois.readLong();
		n = ois.readInt();
	}

	public void reset() {
		ms = 0;
	}

	public Timer start() {
		start = System.currentTimeMillis();
		return this;
	}

	public Timer stop() {
		end = System.currentTimeMillis();
		ms = end - start;
		n = 1;
		return this;
	}

	@Override
	public String toString() {
		long msCopy = ms;
		long m = msCopy / 60000;
		msCopy %= 60000;
		long h = m / 60;
		m %= 60;
		long d = h / 24;
		h %= 24;
		long y = d / 365;
		d %= 365;
		long s = msCopy / 1000;

		StringBuilder sb = new StringBuilder();

		if (y > 0) {
			sb.append(y);
			sb.append('y');
			sb.append(d);
			sb.append('d');
		}
		if (d > 0) {
			sb.append(d);
			sb.append('d');
			sb.append(h);
			sb.append('h');
		} else if (h > 0) {
			sb.append(h);
			sb.append('h');
			sb.append(m);
			sb.append('m');
		} else if (m > 0) {
			sb.append(m);
			sb.append('m');
			sb.append(s);
			sb.append('s');
		} else if (s > 9) {
			sb.append(s);
			sb.append('s');
		} else if (s > 0) {
			sb.append((ms / 100) / 10.0);
			sb.append('s');
		} else {
			sb.append(ms / 1000.0);
			sb.append('s');
		}
		return sb.toString();
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeLong(start);
		oos.writeLong(end);
		oos.writeLong(ms);
		oos.writeInt(n);
	}

}
