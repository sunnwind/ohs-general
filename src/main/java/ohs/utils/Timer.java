package ohs.utils;

/**
 * Simple class for measuring elapsed ms.
 */
public class Timer {

	public static final long[] norms = { 1000, 1000 * 60, 1000 * 60 * 60, 1000 * 60 * 60 * 24 };

	public static Timer newTimer() {
		Timer ret = new Timer();
		ret.start();
		return ret;
	}

	private long start, end, ms;

	private int n;

	private long[] ts = new long[4];

	public Timer() {

	}

	public Timer(long time) {
		start = 0;
		end = time;
		this.ms = time;
	}

	public Timer accumStop() { // Stop and accumulate ms
		end = System.currentTimeMillis();
		ms += end - start;
		n++;
		computeTimes();
		return this;
	}

	public void computeTimes() {
		int i = 0;
		long s = (ms / norms[i++]) % 60;
		long m = (ms / norms[i++]) % 60;
		long h = (ms / norms[i++]) % 24;
		long d = (ms / norms[i++]);

		i = 0;
		ts[i++] = s;
		ts[i++] = m;
		ts[i++] = h;
		ts[i++] = d;
	}

	public long getDays() {
		return ts[3];
	}

	public long getEnd() {
		return end;
	}

	public long getHours() {
		return ts[2];
	}

	public long getMilliseconds() {
		return ms;
	}

	public long getMinute() {
		return ts[1];
	}

	public long getSeconds() {
		return ts[0];
	}

	public long getStart() {
		return start;
	}

	public int getStoppedCnt() {
		return n;
	}

	public long[] getTimes() {
		return ts;
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
		computeTimes();
		return this;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();

		int i = 0;
		long s = ts[i++];
		long m = ts[i++];
		long h = ts[i++];
		long d = ts[i++];

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
		} else {
			sb.append(ms / 1000.0);
			sb.append('s');
		}

		return sb.toString();
	}

}
