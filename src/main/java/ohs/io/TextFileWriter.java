package ohs.io;

import java.io.File;
import java.io.IOException;
import java.io.Writer;

import ohs.utils.Timer;

/**
 * 
 * @author Heung-Seon Oh
 * @version 1.0
 * @date 2009. 4. 14
 * 
 */

public class TextFileWriter {

	private int num_writes;

	private Timer timer;

	private Writer writer;

	public TextFileWriter(File file) {
		this(file.getPath(), FileUtils.UTF_8, false);
	}

	public TextFileWriter(String fileName) {
		this(fileName, FileUtils.UTF_8, false);
	}

	public TextFileWriter(String fileName, String encoding, boolean append) {
		try {
			writer = FileUtils.openBufferedWriter(fileName, encoding, append);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public TextFileWriter(Writer writer) {
		this.writer = writer;
	}

	public void close() {
		try {
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	protected void finalize() throws Throwable {
		super.finalize();
		close();
	}

	public void print(int amount) {
		if (timer.getStart() == 0) {
			timer.start();
		}

		if (num_writes % amount == 0) {
			System.out.print(String.format("\r[%d writes, %s]", num_writes, timer.stop()));
		}
	}

	public void printLast() {
		System.out.println(String.format("\r[%d writes, %s]", num_writes, timer.stop()));
	}

	public void write(String text) {
		try {
			writer.write(text);
			writer.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
