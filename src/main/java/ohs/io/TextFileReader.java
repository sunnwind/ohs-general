package ohs.io;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.List;

import ohs.utils.Generics;
import ohs.utils.Timer;

/**
 * 
 * @author Heung-Seon Oh
 * @version 1.0
 * @date 2009. 4. 14
 * 
 */
public class TextFileReader {

	private String line;

	private long max_lines;

	private long max_nexts;

	private long line_cnt;

	private long next_cnt;

	boolean print_nexts;

	private BufferedReader reader;

	private Timer timer;

	private int print_size = 10000;

	public TextFileReader(File file) {
		this(file.getPath(), FileUtils.UTF_8);
	}

	public TextFileReader(String fileName) {
		this(fileName, FileUtils.UTF_8);
	}

	public TextFileReader(String fileName, String encoding) {
		try {
			reader = FileUtils.openBufferedReader(fileName, encoding);
		} catch (Exception e) {
			e.printStackTrace();
		}

		line = null;
		line_cnt = 0;
		next_cnt = 0;
		timer = new Timer();
		print_nexts = false;

		max_nexts = Integer.MAX_VALUE;
		max_lines = Integer.MAX_VALUE;
	}

	public void close() {
		try {
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	protected void finalize() throws Throwable {
		super.finalize();
		close();
	}

	public BufferedReader getBufferedReader() {
		return reader;
	}

	public long getLineCnt() {
		return line_cnt;
	}

	public long getNextCnt() {
		return next_cnt;
	}

	public List<String> getNextLines() {
		List<String> ret = Generics.newLinkedList();
		do {
			if (next() == null || next().equals("")) {
				break;
			} else {
				ret.add(next());
			}
		} while (hasNext());

		next_cnt++;

		return Generics.newArrayList(ret);
	}

	public Timer getStopWatch() {
		return timer;
	}

	public boolean hasNext() {
		boolean ret = true;

		if (next_cnt > max_nexts || line_cnt > max_lines) {
			ret = false;
		} else {
			line = null;
			try {
				line = reader.readLine();
			} catch (IOException e) {
				e.printStackTrace();
			}

			if (line == null) {
				ret = false;
			} else {
				line_cnt++;
			}
		}
		return ret;
	}

	public String next() {
		return line;
	}

	public void printProgress() {
		if (timer.getStart() == 0) {
			timer.start();
		}

		long remain = 0;

		if (print_nexts) {
			remain = next_cnt % print_size;
		} else {
			remain = line_cnt % print_size;
		}

		if (remain == 0 || line == null) {
			if (print_nexts) {
				System.out.print(String.format("[%d nexts, %d lines, %s]\n", next_cnt, line_cnt, timer.stop()));
			} else {
				System.out.print(String.format("[%s lines, %s]\n", line_cnt, timer.stop()));
			}
		}
	}

	public void setMaxLines(int max_read_lines) {
		this.max_lines = max_read_lines;
	}

	public void setMaxNexts(int max_read_nexts) {
		this.max_nexts = max_read_nexts;
	}

	public void setPrintNexts(boolean print_nexts) {
		this.print_nexts = print_nexts;
	}

	public void setPrintSize(int print_size) {
		this.print_size = print_size;
	}

}
