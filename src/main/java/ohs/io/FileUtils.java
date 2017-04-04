package ohs.io;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.RandomAccessFile;
import java.io.Reader;
import java.io.Writer;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.apache.commons.compress.compressors.CompressorInputStream;
import org.apache.commons.compress.compressors.CompressorOutputStream;
import org.apache.commons.compress.compressors.CompressorStreamFactory;
import org.apache.commons.compress.compressors.gzip.GzipCompressorOutputStream;

import ohs.types.generic.BidMap;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListMap;
import ohs.types.generic.SetMap;
import ohs.types.number.DoubleArray;
import ohs.types.number.LongArray;
import ohs.utils.ByteSize;
import ohs.utils.ByteSize.Type;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

/**
 * @author Heung-Seon Oh
 * @version 1.2
 * @date 2009. 5. 10
 * 
 */
public class FileUtils {

	public static final String UTF_8 = "UTF-8";

	public static final String EUC_KR = "euc-kr";

	public static final String LINE_SIZE = "###LINES###";

	public static final int DEFAULT_BUF_SIZE = (int) new ByteSize(64, Type.MEGA).getBytes();

	private static void addFilesUnder(File root, List<File> files, boolean recursive) {
		if (root != null) {
			File[] children = root.listFiles();
			if (children != null) {
				for (File child : root.listFiles()) {
					if (child.isFile()) {
						files.add(child);
					} else {
						if (recursive) {
							addFilesUnder(child, files, recursive);
						}
					}
				}
			}
		}
	}

	public static File appendFileNameSuffix(File file, String suffix) {
		String filePath = getCanonicalPath(file);
		if (!filePath.endsWith(suffix)) {
			filePath += suffix;
			file = new File(filePath);
		}
		return file;
	}

	private static void compress(File root, File input, TarArchiveOutputStream taos) throws IOException {

		if (input.isFile()) {
			System.out.println("Adding File: " + root.toURI().relativize(input.toURI()).getPath());

			BufferedInputStream bis = new BufferedInputStream(new FileInputStream(input));

			/** Step: 3 ---> Create a tar entry for each file that is read. **/

			/**
			 * relativize is used to to add a file to a tar, without including the entire path from root.
			 **/

			TarArchiveEntry tae = new TarArchiveEntry(input, root.getParentFile().toURI().relativize(input.toURI()).getPath());

			/** Step: 4 ---> Put the tar entry using putArchiveEntry. **/

			taos.putArchiveEntry(tae);

			/**
			 * Step: 5 ---> Write the data to the tar file and close the input stream.
			 **/

			int count;
			byte data[] = new byte[2048];
			while ((count = bis.read(data, 0, 2048)) != -1) {
				taos.write(data, 0, count);
			}
			bis.close();

			/** Step: 6 --->close the archive entry. **/

			taos.closeArchiveEntry();

		} else {
			if (input.listFiles() != null) {
				/** Add an empty folder to the tar **/
				if (input.listFiles().length == 0) {

					System.out.println("Adding Empty Folder: " + root.toURI().relativize(input.toURI()).getPath());
					TarArchiveEntry entry = new TarArchiveEntry(input, root.getParentFile().toURI().relativize(input.toURI()).getPath());
					taos.putArchiveEntry(entry);
					taos.closeArchiveEntry();
				}

				for (File file : input.listFiles())
					compress(root, file, taos);
			}
		}
	}

	public static void compress(String inPath, String outFileName) throws Exception {

		if (!outFileName.endsWith(".tar.gz")) {
			outFileName += ".tar.gz";
		}

		/** Step: 1 ---> create a TarArchiveOutputStream object. **/
		TarArchiveOutputStream taos = new TarArchiveOutputStream(
				new GzipCompressorOutputStream(new BufferedOutputStream(new FileOutputStream(outFileName))));

		// TAR has an 8 gig file limit by default, this gets around that
		taos.setBigNumberMode(TarArchiveOutputStream.BIGNUMBER_STAR); // to get
																		// past
																		// the 8
																		// gig
																		// limit
		// TAR originally didn't support long file names, so enable the support
		// for it
		taos.setLongFileMode(TarArchiveOutputStream.LONGFILE_GNU);

		/**
		 * Step: 2 --->Open the source data and get a list of files from given directory recursively.
		 **/

		File input = new File(inPath);

		compress(input.getParentFile(), input, taos);

		/** Step: 7 --->close the output stream. **/

		taos.close();

		System.out.println("tar.gz file created successfully!!");

	}

	// public static ByteArray readByteArrayMatrix(FileChannel fc, int size)
	// throws Exception {
	// return new ByteArray(readByteBuffer(fc, size).array());
	// }

	public static void copy(String inFileName, String outDirName) throws Exception {
		createFolder(outDirName);

		InputStream is = null;
		OutputStream os = null;
		is = new FileInputStream(inFileName);
		os = new FileOutputStream(outDirName);
		byte[] buffer = new byte[1024];
		int length;
		while ((length = is.read(buffer)) > 0) {
			os.write(buffer, 0, length);
		}
		is.close();
		os.close();
	}

	public static void copyFolder(String inDir, String outDir) throws Exception {
		for (File inFile : getFilesUnder(inDir)) {
			String path = inFile.getPath();
			path = path.replace(inDir, outDir);
			copy(inFile.getPath(), path);
		}
	}

	public static int countLines(String fileName) throws Exception {
		int ret = 0;

		BufferedReader reader = openBufferedReader(fileName);
		String line = reader.readLine();

		if (line != null && line.startsWith(LINE_SIZE)) {
			ret = Integer.parseInt(line.split("\t")[1]);
		} else {
			if (line != null) {
				ret++;
				while ((line = reader.readLine()) != null) {
					ret++;
				}
			}
		}
		reader.close();
		return ret;
	}

	// public static ByteArrayMatrix readByteArrayMatrix(FileChannel fc) throws
	// Exception{
	// int size = readInteger(fc);
	// }

	public static int countLinesUnder(String dirName) throws Exception {
		return countLinesUnderHere(new File(dirName));
	}

	private static int countLinesUnderHere(File root) throws Exception {
		int ret = 0;
		if (root != null) {
			File[] children = root.listFiles();
			if (children != null) {
				for (File child : root.listFiles()) {
					if (child.isFile()) {
						ret += countLines(child.getPath());
					} else {
						ret += countLinesUnderHere(child);
					}
				}
			}
		}
		return ret;
	}

	public static void createFolder(File file) {
		String sep = System.getProperty("file.separator");

		if (sep.equals("\\")) {
			sep = "\\\\";
		}

		if (!file.exists()) {
			// if (file.isDirectory()) {
			file.mkdirs();
			// }

			// else {
			// if (file.getPath().split(sep).length > 1) {
			// File parentFile = new File(file.getParent());
			// if (!parentFile.exists()) {
			// parentFile.mkdirs();
			// }
			// }
			// }
		}
	}

	public static void createFolder(String fileName) {
		createFolder(new File(fileName));
	}

	public static boolean delete(String fileName) {
		File f = new File(fileName);
		boolean ret = false;
		if (f.exists()) {
			ret = f.delete();
		}
		return ret;
	}

	private static int deleteFiles(File root) {
		int file_cnt = 0;
		if (root.exists()) {
			if (root.isDirectory()) {
				for (File child : root.listFiles()) {
					file_cnt += deleteFiles(child);
				}
				root.delete();
			} else if (root.isFile()) {
				root.delete();
				file_cnt++;
			}
		}
		return file_cnt;
	}

	public static void deleteFilesUnder(File dir) {
		int num_files = deleteFiles(dir);
		System.out.println(String.format("delete [%d] files at [%s]", num_files, dir.getPath()));
	}

	public static void deleteFilesUnder(String dirName) {
		deleteFilesUnder(new File(dirName));
	}

	public static boolean exists(String fileName) {
		return new File(fileName).exists();
	}

	public static String getCanonicalPath(File file) {
		String ret = null;
		try {
			ret = file.getCanonicalPath();
			ret = ret.replace("\\", "/");
		} catch (IOException e) {
			e.printStackTrace();
		}
		return ret;
	}

	public static String getExtension(String fileName) {
		int idx = fileName.lastIndexOf(".");
		if (idx > 0) {
			fileName = fileName.substring(idx + 1);
		}
		return fileName;
	}

	public static String getFileName(File file) {
		return removeExtension(file.getName());
	}

	public static List<File> getFilesUnder(File dir) {
		return getFilesUnder(dir, true);
	}

	private static List<File> getFilesUnder(File dir, boolean recursive) {
		List<File> files = Generics.newLinkedList();
		addFilesUnder(dir, files, recursive);

		Collections.sort(files);

		long bytes = 0;
		for (int i = 0; i < files.size(); i++) {
			bytes += files.get(i).length();
		}
		ByteSize bs = new ByteSize(bytes);

		System.out.println(String.format("read [%d] files (%s) at [%s]", files.size(), bs.toString(), dir.getPath()));
		return files;
	}

	public static List<File> getFilesUnder(String dirName) {
		return getFilesUnder(new File(dirName), true);
	}

	public static long length(String fileName) {
		return new File(fileName).length();
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// compress("../../data/news_ir/content_nlp",
		// "../../data/news_ir/test.tar.gz");

		{

			File f = new File("./test.ser");
			f.delete();

			FileChannel fc = openFileChannel("./test.ser", "rw");

			int size = 100000000;

			ByteArrayMatrix m = new ByteArrayMatrix(2);

			m.add(new ByteArray(new byte[size]));
			m.add(new ByteArray(new byte[size]));

			long size1 = ByteArrayUtils.sizeOfByteBuffer(m);

			System.out.println(new ByteSize(size1));

			long[] info = write(m, fc);

			fc.close();

			System.out.println(new LongArray(info));

			System.out.println(new ByteSize(info[1]));

			fc = openFileChannel("./test.ser", "rw");

			ByteArrayMatrix m2 = readByteArrayMatrix(fc);

			long size2 = ByteArrayUtils.sizeOfByteBuffer(m2);

			System.out.println(new ByteSize(size2));

			fc.close();

		}

		// String s = "ABCDE";
		//
		// {
		// ObjectOutputStream oos =
		// openObjectOutputStream("../../data/entity_iden/wiki/test1.ser");
		// oos.write(s.getBytes());
		// oos.close();
		// }
		//
		// {
		// ObjectOutputStream oos =
		// openObjectOutputStream("../../data/entity_iden/wiki/test2.ser");
		//
		// for (int i = 0; i < 1; i++) {
		// oos.writeUTF(s);
		// }
		//
		// oos.close();
		// }
		//
		// {
		// ObjectOutputStream oos =
		// openObjectOutputStream("../../data/entity_iden/wiki/test3.ser");
		// write(oos, s);
		//
		// oos.close();
		// }
		//
		// {
		// ObjectInputStream ois =
		// openObjectInputStream("../../data/entity_iden/wiki/test2.ser");
		//
		// for (int i = 0; i < 1; i++) {
		// String ss = ois.readUTF();
		// System.out.println(ss);
		// }
		// }
		//
		// {
		// double[] ar = ArrayUtils.range(10000000, 0.0, 1);
		// ObjectOutputStream oos =
		// openObjectOutputStream("../../data/entity_iden/wiki/test-a1.ser.gz");
		//
		// FileUtils.writeDoubles(oos, ar);
		// oos.close();
		//
		// }
		//
		// {
		// int[] ar = ArrayUtils.range(10, 0, 1);
		// ObjectOutputStream oos =
		// openObjectOutputStream("../../data/entity_iden/wiki/test-a2.ser.gz");
		//
		// FileUtils.writeIntegers(oos, ar);
		// oos.close();
		// }

		System.out.println("process ends.");
	}

	public static void move(String inFileName, String outDirName) throws Exception {
		copy(inFileName, outDirName);
		new File(inFileName).delete();
	}

	public static void moveFolder(String srcDir, String desDir) throws Exception {
		List<File> files = getFilesUnder(srcDir);
		for (File inputFile : files) {
			String path = inputFile.getPath();
			path = path.replace(srcDir, desDir);
			copy(inputFile.getPath(), path);
		}

		for (File file : files) {
			file.delete();
		}
	}

	public static BufferedReader openBufferedReader(String fileName) throws Exception {
		return openBufferedReader(fileName, UTF_8);
	}

	public static BufferedReader openBufferedReader(String fileName, String encoding) throws Exception {
		File file = new File(fileName);
		FileInputStream fis = new FileInputStream(file);
		InputStreamReader isr = null;

		if (file.getName().endsWith(".gz")) {
			CompressorInputStream cis = new CompressorStreamFactory().createCompressorInputStream(CompressorStreamFactory.GZIP, fis);
			isr = new InputStreamReader(cis, encoding);
		} else if (file.getName().endsWith(".zip")) {
			isr = new InputStreamReader(new ZipInputStream(fis));
		} else if (file.getName().endsWith(".bz2")) {
			// byte[] ignoreBytes = new byte[2];
			// fis.read(ignoreBytes); // "B", "Z" bytes from commandline tools
			// ret = new BufferedReader(new InputStreamReader(new
			// CBZip2InputStream(fis)));
			CompressorInputStream cis = new CompressorStreamFactory().createCompressorInputStream(CompressorStreamFactory.BZIP2, fis);
			isr = new InputStreamReader(cis, encoding);
		} else {
			isr = new InputStreamReader(fis, encoding);
		}
		return new BufferedReader(isr);
	}

	public static BufferedWriter openBufferedWriter(String fileName) throws Exception {
		return openBufferedWriter(fileName, UTF_8, false);
	}

	public static BufferedWriter openBufferedWriter(String fileName, String encoding, boolean append) throws Exception {
		String fileSeparator = System.getProperty("file.separator");

		if (fileSeparator.equals("\\")) {
			fileSeparator = "\\\\";
		}

		File file = new File(fileName);

		if (file.getPath().split(fileSeparator).length > 1) {
			File parentFile = new File(file.getParent());
			if (!parentFile.exists()) {
				parentFile.mkdirs();
			}
		}

		if (!file.exists()) {
			append = false;
		}

		FileOutputStream fos = new FileOutputStream(file, append);
		OutputStreamWriter osw = null;

		if (file.getName().endsWith(".gz")) {
			// osw = new OutputStreamWriter(new GZIPOutputStream(new
			// FileOutputStream(file, append)), encoding);
			CompressorOutputStream cos = new CompressorStreamFactory().createCompressorOutputStream(CompressorStreamFactory.GZIP, fos);
			osw = new OutputStreamWriter(cos, encoding);
		} else if (file.getName().endsWith(".bz2")) {
			// osw = new OutputStreamWriter(new CBZip2OutputStream(new
			// FileOutputStream(file, append)), encoding);
			CompressorOutputStream cos = new CompressorStreamFactory().createCompressorOutputStream(CompressorStreamFactory.BZIP2, fos);
			osw = new OutputStreamWriter(cos, encoding);
		} else {
			osw = new OutputStreamWriter(fos, encoding);
		}

		return new BufferedWriter(osw);
	}

	public static FileChannel openFileChannel(File file, String mode) throws Exception {
		if (!file.getParentFile().exists()) {
			file.getParentFile().mkdirs();
		}

		return new RandomAccessFile(file, mode).getChannel();
	}

	public static FileChannel openFileChannel(String fileName, String mode) throws Exception {
		return openFileChannel(new File(fileName), mode);
	}

	public static ObjectInputStream openObjectInputStream(String fileName) throws Exception {
		ObjectInputStream ret = null;
		if (fileName.endsWith(".gz")) {
			ret = new ObjectInputStream(new GZIPInputStream(new FileInputStream(fileName)));
		} else if (fileName.endsWith(".zip")) {
			ret = new ObjectInputStream(new ZipInputStream(new FileInputStream(fileName)));
		} else if (fileName.endsWith(".ser")) {
			ret = new ObjectInputStream(new FileInputStream(fileName));
		}
		return ret;
	}

	public static ObjectOutputStream openObjectOutputStream(String fileName) throws Exception {
		return openObjectOutputStream(fileName, false);
	}

	public static ObjectOutputStream openObjectOutputStream(String fileName, boolean append) throws Exception {
		File file = new File(fileName);
		File parent = file.getParentFile();
		if (!parent.exists()) {
			parent.mkdirs();
		}

		ObjectOutputStream ret = null;
		if (file.getName().endsWith(".ser.gz")) {
			ret = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(file, append)));
		} else if (file.getName().endsWith(".zip")) {
			ret = new ObjectOutputStream(new ZipOutputStream(new FileOutputStream(file, append)));
		} else if (file.getName().endsWith(".ser")) {
			ret = new ObjectOutputStream(new FileOutputStream(file, append));
		}
		return ret;
	}

	public static InputStreamReader openUrl(URL url) throws IOException {
		URLConnection urlConn = url.openConnection();
		urlConn.setRequestProperty("User-agent", "Mozilla/4.0");

		HttpURLConnection httpUrlConn = (HttpURLConnection) urlConn;
		httpUrlConn.setConnectTimeout(2000);
		int responseCode = httpUrlConn.getResponseCode();

		if (responseCode == HttpURLConnection.HTTP_OK) {
			return new InputStreamReader(urlConn.getInputStream(), "UTF-8");
		} else {
			throw new IOException();
		}
	}

	public static ByteArray readByteArray(FileChannel fc) throws Exception {
		return new ByteArray(readByteBuffer(fc).array());
	}

	public static ByteArray readByteArray(FileChannel fc, int size) throws Exception {
		return new ByteArray(readByteBuffer(fc, size).array());
	}

	public static ByteArray readByteArray(ObjectInputStream ois) throws Exception {
		return readByteArray(ois, ois.readInt());
	}

	public static ByteArray readByteArray(ObjectInputStream ois, int size) throws Exception {
		byte[] ret = new byte[size];
		ois.read(ret);
		return new ByteArray(ret);
	}

	public static ByteArrayMatrix readByteArrayMatrix(FileChannel a) throws Exception {
		int size = readInteger(a);
		ByteArrayMatrix ret = new ByteArrayMatrix(size);
		for (int i = 0; i < size; i++) {
			ret.add(readByteArray(a));
		}
		return ret;
	}

	public static ByteArrayMatrix readByteArrayMatrix(ObjectInputStream ois) throws Exception {
		return ByteArrayUtils.toByteArrayMatrix(readByteArray(ois));
	}

	public static List<ByteArrayMatrix> readByteArrayMatrixList(FileChannel fc) throws Exception {
		int size = readInteger(fc);
		List<ByteArrayMatrix> ret = Generics.newLinkedList();
		for (int i = 0; i < size; i++) {
			ret.add(readByteArrayMatrix(fc));
		}
		return ret;
	}

	public static ByteBuffer readByteBuffer(FileChannel fc) throws Exception {
		return readByteBuffer(fc, readInteger(fc));
	}

	public static ByteBuffer readByteBuffer(FileChannel fc, int size) throws Exception {
		long pos_old = fc.position();

		ByteBuffer buf = ByteBuffer.allocate(size);
		fc.read(buf);
		buf.flip();

		long pos_new = fc.position();
		if (pos_new == pos_old) {
			fc.position(pos_old + size);
		}

		return buf;
	}

	public static double[] readDoubleArray(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		double[] ret = new double[size];
		for (int i = 0; i < size; i++) {
			ret[i] = ois.readDouble();
		}
		return ret;
	}

	public static List<Double> readDoubleList(ObjectInputStream ois) throws Exception {
		List<Double> ret = Generics.newLinkedList();
		int size = ois.readInt();
		for (int i = 0; i < size; i++) {
			ret.add(ois.readDouble());
		}
		return ret;
	}

	public static double[][] readDoubleMatrix(ObjectInputStream ois) throws Exception {
		int rowSize = ois.readInt();
		double[][] ret = new double[rowSize][];
		for (int i = 0; i < rowSize; i++) {
			ret[i] = readDoubleArray(ois);
		}
		return ret;
	}

	public static String readFromText(File file) throws Exception {
		return readFromText(file.getCanonicalPath(), UTF_8);
	}

	public static String readFromText(Reader reader) throws Exception {
		StringBuffer sb = new StringBuffer();
		while (true) {
			int i = reader.read();
			if (i == -1) {
				break;
			} else {
				sb.append((char) i);
			}
		}
		return sb.toString();
	}

	public static String readFromText(String fileName) throws Exception {
		return readFromText(fileName, UTF_8);
	}

	public static String readFromText(String fileName, String encoding) throws Exception {
		StringBuffer ret = new StringBuffer();
		BufferedReader reader = openBufferedReader(fileName, encoding);
		ret.append(readFromText(reader));
		reader.close();
		return ret.toString();
	}

	public static int readInteger(FileChannel fc) throws Exception {
		ByteBuffer buf = readByteBuffer(fc, Integer.BYTES);
		int ret = buf.getInt();
		return ret;
	}

	public static Counter<Integer> readIntegerCounter(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		Counter<Integer> ret = Generics.newCounter(size);
		for (int i = 0; i < size; i++) {
			ret.setCount(ois.readInt(), ois.readDouble());
		}
		return ret;
	}

	public static List<Integer> readIntegerList(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		List<Integer> ret = Generics.newArrayList(size);
		for (int i = 0; i < size; i++) {
			ret.add(ois.readInt());
		}
		return ret;
	}

	public static ListMap<Integer, Integer> readIntegerListMap(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		ListMap<Integer, Integer> ret = Generics.newListMap(size);
		for (int i = 0; i < size; i++) {
			ret.put(ois.readInt(), readIntegerList(ois));
		}
		return ret;
	}

	public static Map<Integer, Integer> readIntegerMap(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		Map<Integer, Integer> ret = Generics.newHashMap(size);
		for (int i = 0; i < size; i++) {
			ret.put(ois.readInt(), ois.readInt());
		}
		// System.out.printf("read [%d] ents.\n", ret.size());
		return ret;
	}

	public static int[][] readIntegerMatrix(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		int[][] ret = new int[size][];
		for (int i = 0; i < size; i++) {
			ret[i] = readIntegers(ois);
		}
		return ret;
	}

	public static int[][] readIntegerMatrix(String fileName) throws Exception {
		System.out.printf("read at [%s].\n", fileName);
		ObjectInputStream ois = openObjectInputStream(fileName);
		int[][] ret = readIntegerMatrix(ois);
		ois.close();
		return ret;
	}

	public static int[][][] readIntegerMatrix3D(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		int[][][] ret = new int[size][][];
		for (int i = 0; i < size; i++) {
			ret[i] = readIntegerMatrix(ois);
		}
		return ret;
	}

	public static int[][][] readIntegerMatrix3D(String fileName) throws Exception {
		System.out.printf("read at [%s].\n", fileName);
		ObjectInputStream ois = openObjectInputStream(fileName);
		int[][][] ret = readIntegerMatrix3D(ois);
		ois.close();
		return ret;
	}

	public static int[] readIntegers(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		int[] ret = new int[size];
		for (int i = 0; i < size; i++) {
			ret[i] = ois.readInt();
		}
		return ret;
	}

	public static int[] readIntegers(String fileName) throws Exception {
		System.out.printf("read at [%s].\n", fileName);
		ObjectInputStream ois = openObjectInputStream(fileName);
		int[] ret = readIntegers(ois);
		ois.close();
		return ret;
	}

	public static Set<Integer> readIntegerSet(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		Set<Integer> ret = Generics.newHashSet(size);
		for (int i = 0; i < size; i++) {
			ret.add(ois.readInt());
		}
		return ret;
	}

	public static SetMap<Integer, Integer> readIntegerSetMap(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		SetMap<Integer, Integer> ret = Generics.newSetMap(size);
		for (int i = 0; i < size; i++) {
			ret.put(ois.readInt(), readIntegerSet(ois));
		}
		return ret;
	}

	public static BidMap<Integer, String> readIntegerStringBidMap(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		BidMap<Integer, String> ret = Generics.newBidMap(size);
		for (int i = 0; i < size; i++) {
			ret.put(ois.readInt(), ois.readUTF());
		}
		return ret;
	}

	public static BidMap<Integer, String> readIntegerStringBidMap(String fileName) throws Exception {
		ObjectInputStream ois = openObjectInputStream(fileName);
		BidMap<Integer, String> ret = readIntegerStringBidMap(ois);
		ois.close();
		return ret;
	}

	public static Map<Integer, String> readIntegerStringMap(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		Map<Integer, String> ret = Generics.newHashMap(size);
		for (int i = 0; i < size; i++) {
			ret.put(ois.readInt(), ois.readUTF());
		}
		return ret;
	}

	public static List<String> readLinesFromText(BufferedReader reader, int size) throws Exception {
		List<String> ret = Generics.newLinkedList();
		String line = reader.readLine();
		if (line.startsWith(LINE_SIZE)) {
			String[] parts = line.split("\t");
			size = Integer.parseInt(parts[1]);
			ret = Generics.newArrayList(size);
		} else {
			ret.add(line);
		}

		for (int i = 0; i < size; i++) {
			line = reader.readLine();
			if (line == null) {
				break;
			}
			ret.add(line);
		}
		return ret;
	}

	public static List<String> readLinesFromText(File file) throws Exception {
		return readLinesFromText(file.getPath());
	}

	public static List<String> readLinesFromText(String fileName) throws Exception {
		return readLinesFromText(fileName, UTF_8, Integer.MAX_VALUE);
	}

	public static List<String> readLinesFromText(String fileName, int size) throws Exception {
		return readLinesFromText(fileName, UTF_8, size);
	}

	public static List<String> readLinesFromText(String fileName, String encoding) throws Exception {
		return readLinesFromText(fileName, encoding, Integer.MAX_VALUE);
	}

	public static List<String> readLinesFromText(String fileName, String encoding, int size) throws Exception {
		BufferedReader reader = openBufferedReader(fileName, encoding);
		List<String> ret = readLinesFromText(reader, size);
		reader.close();

		System.out.printf("read [%d] lines at [%s]\n", ret.size(), fileName);
		return ret;
	}

	public static long[] readLongArray(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		long[] ret = new long[size];
		for (int i = 0; i < size; i++) {
			ret[i] = ois.readLong();
		}
		return ret;
	}

	public static long[] readLongArray(String fileName) throws Exception {
		System.out.printf("read at [%s].\n", fileName);
		ObjectInputStream ois = openObjectInputStream(fileName);
		long[] ret = readLongArray(ois);
		ois.close();
		return ret;
	}

	public static String[] readStringArray(ObjectInputStream ois) throws Exception {
		String[] ret = new String[ois.readInt()];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = ois.readUTF();
		}
		return ret;
	}

	public static Counter<String> readStringCounter(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		Counter<String> ret = new Counter<String>(size);
		for (int i = 0; i < size; i++) {
			ret.setCount(ois.readUTF(), ois.readDouble());
		}
		return ret;
	}

	public static Counter<String> readStringCounterFromText(String fileName) throws Exception {

		BufferedReader br = openBufferedReader(fileName);
		String line = br.readLine();

		Counter<String> ret = new Counter<String>();

		if (line.startsWith(LINE_SIZE)) {
			int size = Integer.parseInt(line.split("\t")[1]);
			ret = Generics.newCounter(size);
		}

		while ((line = br.readLine()) != null) {
			String[] parts = line.split("\t");
			int len = parts.length;

			if (len == 1) {
				ret.setCount(parts[0], 1);
			} else if (len == 2) {
				ret.setCount(parts[0], Double.parseDouble(parts[1]));
			} else if (len > 2) {
				ret.setCount(StrUtils.join("\t", parts, 0, len - 1), Double.parseDouble(parts[len - 1]));
			}
		}

		br.close();

		System.out.printf("read [%d] ents at [%s]\n", ret.size(), fileName);
		return ret;
	}

	public static CounterMap<String, String> readStringCounterMap(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		CounterMap<String, String> ret = Generics.newCounterMap(size);
		for (int i = 0; i < size; i++) {
			ret.setCounter(ois.readUTF(), readStringCounter(ois));
		}
		return ret;
	}

	public static CounterMap<String, String> readStringCounterMap(String fileName) throws Exception {
		ObjectInputStream ois = openObjectInputStream(fileName);
		CounterMap<String, String> ret = readStringCounterMap(ois);
		ois.close();
		return ret;
	}

	public static CounterMap<String, String> readStringCounterMapFromText(String fileName) throws Exception {
		CounterMap<String, String> ret = Generics.newCounterMap();

		BufferedReader br = openBufferedReader(fileName);

		String line = br.readLine();

		if (line.startsWith(LINE_SIZE)) {
			int size = Integer.parseInt(line.split("\t")[1]);
			ret = Generics.newCounterMap(size);
		}

		int num_entries = 0;

		while ((line = br.readLine()) != null) {
			String[] parts = line.split("\t");
			String outKey = parts[0];
			for (int i = 2; i < parts.length; i++) {
				String[] two = StrUtils.split2Two(":", parts[i]);
				String inKey = two[0];
				double cnt = Double.parseDouble(two[1]);
				ret.setCount(outKey, inKey, cnt);
				num_entries++;
			}
		}
		br.close();

		System.out.printf("read [%d] ents at [%s]\n", num_entries, fileName);
		return ret;
	}

	public static Map<String, String> readStringHashMapFromText(String fileName) throws Exception {
		List<String> lines = readLinesFromText(fileName);
		Map<String, String> m = Generics.newHashMap(lines.size());
		for (String line : lines) {
			String[] parts = line.split("\t");
			m.put(parts[0], parts[1]);
		}
		return m;
	}

	public static Indexer<String> readStringIndexer(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		Indexer<String> ret = new Indexer<String>(size);
		for (int i = 0; i < size; i++) {
			ret.add(ois.readUTF());
		}
		return ret;
	}

	public static Indexer<String> readStringIndexerFromText(BufferedReader br) throws Exception {
		String[] two = br.readLine().split("\t");
		int num_lines = Integer.parseInt(two[1]);
		Indexer<String> ret = Generics.newIndexer();

		for (int i = 0; i < num_lines; i++) {
			ret.add(br.readLine());
		}
		return ret;
	}

	public static Indexer<String> readStringIndexerFromText(String fileName) throws Exception {
		System.out.printf("read [%s].\n", fileName);
		Indexer<String> ret = new Indexer<String>();
		BufferedReader br = openBufferedReader(fileName);
		String line = null;
		while ((line = br.readLine()) != null) {
			ret.add(line);
		}
		br.close();
		return ret;
	}

	public static List<String> readStringList(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		List<String> ret = Generics.newArrayList(size);
		for (int i = 0; i < size; i++) {
			ret.add(ois.readUTF());
		}
		return ret;
	}

	public static Map<String, String> readStringMapFromText(String fileName) throws Exception {
		Map<String, String> ret = new HashMap<String, String>();
		for (String line : readLinesFromText(fileName)) {
			String[] parts = line.split("\t");
			if (parts.length != 2) {
				throw new Exception("# parts is not 2.");
			}
			ret.put(parts[0], parts[1]);
		}
		return ret;
	}

	public static HashSet<String> readStringSetFromText(String fileName) throws Exception {
		return new HashSet<String>(readLinesFromText(fileName));
	}

	public static String removeExtension(String fileName) {
		int end = fileName.lastIndexOf(".");
		if (end > 0) {
			fileName = fileName.substring(0, end);
		}
		return fileName;
	}

	public static long[] write(ByteArray a, ByteBufferWrapper b, FileChannel c) throws Exception {
		int size = (int) ByteArrayUtils.sizeOfByteBuffer(a);
		long[] info = null;

		if (size > b.capacity()) {
			long start = c.position();
			long end = start;

			b.write(a.size());

			int s = 0;
			long len1 = 0;

			while (s < a.size()) {
				int e = Math.min(a.size(), s + b.capacity() - b.position());
				for (int j = s; j < e; j++) {
					b.write(a.value(j));
				}
				long[] tmp = write(b.getByteBuffer(), c);
				len1 += tmp[1];
				s = e;
			}

			end = c.position();

			long len2 = end - start;

			info = new long[] { start, end - start };
		} else {
			b.write(a);
			info = write(b.getByteBuffer(), c);
		}

		return info;
	}

	public static long[] write(ByteArray a, FileChannel b) throws Exception {
		int size = (int) ByteArrayUtils.sizeOfByteBuffer(a);
		ByteBufferWrapper c = new ByteBufferWrapper(Math.min(size, DEFAULT_BUF_SIZE));
		return write(a, c, b);
	}

	public static long[] write(ByteArrayMatrix a, ByteBufferWrapper b, FileChannel c) throws Exception {
		long[] info = null;
		long size = ByteArrayUtils.sizeOfByteBuffer(a);

		if (size > b.capacity()) {
			long start = c.position();
			long end = start;

			long len1 = 0;
			long len2 = 0;

			b.write(a.size());

			for (ByteArray d : a) {
				long[] tmp = write(d, b, c);
				len1 += tmp[1];
			}

			end = c.position();
			len2 = end - start;

			info = new long[] { start, len2 };
		} else {
			b.write(a);
			info = write(b.getByteBuffer(), c);
		}
		return info;
	}

	public static long[] write(ByteArrayMatrix a, FileChannel b) throws Exception {
		long size = ByteArrayUtils.sizeOfByteBuffer(a);
		ByteBufferWrapper buf = new ByteBufferWrapper((int) Math.min(size, DEFAULT_BUF_SIZE));
		return write(a, buf, b);
	}

	public static long[] write(ByteBuffer a, FileChannel b) throws Exception {
		long pos_old = b.position();
		int size = a.position();

		a.flip();
		b.write(a);
		a.clear();

		long pos_new = b.position();

		if (pos_old == pos_new) {
			b.position(pos_old + size);
		}
		// System.out.printf("%d, %d\n", pos_old, pos_new);
		return new long[] { pos_old, size };
	}

	public static void write(ObjectOutputStream oos, ByteArray a) throws Exception {
		oos.writeInt(a.size());
		if (a.size() == a.length()) {
			oos.write(a.values());
		} else {
			oos.write(a.subArray(0, a.size()).values());
		}
	}

	public static void write(ObjectOutputStream oos, ByteArrayMatrix a) throws Exception {
		write(oos, ByteArrayUtils.toByteArray(a));
	}

	public static void write(String fileName, boolean[] x) throws Exception {
		ObjectOutputStream oos = openObjectOutputStream(fileName);
		writeBooleans(oos, x);
		oos.close();
	}

	public static void writeAsText(String fileName, boolean append, String text) throws Exception {
		writeAsText(fileName, UTF_8, append, text);
	}

	public static void writeAsText(String fileName, String text) throws Exception {
		writeAsText(fileName, UTF_8, false, text);
	}

	public static void writeAsText(String fileName, String encoding, boolean append, String text) throws Exception {
		Writer writer = openBufferedWriter(fileName, encoding, append);
		writer.write(text);
		writer.flush();
		writer.close();
	}

	public static void writeBooleans(ObjectOutputStream oos, boolean[] x) throws IOException {
		oos.writeInt(x.length);
		for (int i = 0; i < x.length; i++) {
			oos.writeBoolean(x[i]);
		}
		// oos.flush();
	}

	public static void writeDoubleArray(ObjectOutputStream oos, DoubleArray a) throws Exception {
		// ByteArray b = ByteArrayUtils.toByteArray(a);
		// oos.write(b.values());
	}

	public static void writeDoubleCollection(ObjectOutputStream oos, Collection<Double> c) throws Exception {
		oos.writeInt(c.size());
		Iterator<Double> iter = c.iterator();
		while (iter.hasNext()) {
			oos.writeDouble(iter.next());
		}
		// oos.flush();
	}

	public static void writeDoubleMatrix(ObjectOutputStream oos, double[][] x) throws Exception {
		oos.writeInt(x.length);
		for (int i = 0; i < x.length; i++) {
			writeDoubles(oos, x[i]);
		}
		oos.flush();
	}

	public static void writeDoubles(ObjectOutputStream oos, double[] x) throws Exception {
		int size = x.length;
		oos.writeInt(size);
		for (int i = 0; i < x.length; i++) {
			oos.writeDouble(x[i]);
		}
		// oos.flush();
	}

	public static void writeIntegerCollection(ObjectOutputStream oos, Collection<Integer> c) throws Exception {
		oos.writeInt(c.size());
		Iterator<Integer> iter = c.iterator();
		while (iter.hasNext()) {
			oos.writeInt(iter.next());
		}
		// oos.flush();
	}

	public static void writeIntegerCounter(ObjectOutputStream oos, Counter<Integer> c) throws Exception {
		oos.writeInt(c.size());
		for (Entry<Integer, Double> e : c.entrySet()) {
			oos.writeInt(e.getKey());
			oos.writeDouble(e.getValue());
		}
		oos.flush();
	}

	public static void writeIntegerDoublePairs(ObjectOutputStream oos, int[] indexes, double[] values) throws Exception {
		int size = indexes.length;
		oos.writeInt(size);
		for (int i = 0; i < indexes.length; i++) {
			oos.writeInt(indexes[i]);
			oos.writeDouble(values[i]);
		}
		oos.flush();
	}

	public static void writeIntegerListMap(ObjectOutputStream oos, ListMap<Integer, Integer> lm) throws Exception {
		oos.writeInt(lm.size());

		for (int key : lm.keySet()) {
			oos.writeInt(key);
			writeIntegerCollection(oos, lm.get(key));
		}
	}

	public static void writeIntegerMap(ObjectOutputStream oos, Map<Integer, Integer> m) throws Exception {
		oos.writeInt(m.size());
		for (Integer key : m.keySet()) {
			oos.writeInt(key);
			oos.writeInt(m.get(key));
		}
		oos.flush();
	}

	public static void writeIntegerMatrix(ObjectOutputStream oos, int[][] x) throws Exception {
		oos.writeInt(x.length);
		for (int i = 0; i < x.length; i++) {
			writeIntegers(oos, x[i]);
		}
		oos.flush();
	}

	public static void writeIntegerMatrix(ObjectOutputStream oos, int[][][] x) throws Exception {
		oos.writeInt(x.length);
		for (int i = 0; i < x.length; i++) {
			writeIntegerMatrix(oos, x[i]);
		}
	}

	public static void writeIntegerMatrix(String fileName, int[][] x) throws Exception {
		System.out.printf("write at [%s]\n", fileName);
		ObjectOutputStream oos = openObjectOutputStream(fileName);
		writeIntegerMatrix(oos, x);
		oos.close();
	}

	public static void writeIntegerMatrix(String fileName, int[][][] x) throws Exception {
		System.out.printf("write at [%s]\n", fileName);
		ObjectOutputStream oos = openObjectOutputStream(fileName);
		writeIntegerMatrix(oos, x);
		oos.close();
	}

	public static void writeIntegers(ObjectOutputStream oos, int[] x) throws Exception {
		int size = x.length;
		oos.writeInt(size);
		for (int i = 0; i < x.length; i++) {
			oos.writeInt(x[i]);
		}
		oos.flush();
	}

	public static void writeIntegers(String fileName, int[] x) throws Exception {
		ObjectOutputStream oos = openObjectOutputStream(fileName);
		writeIntegers(oos, x);
		oos.close();
	}

	public static void writeIntegerSet(ObjectOutputStream oos, Set<Integer> s) throws Exception {
		oos.writeInt(s.size());
		for (int value : s) {
			oos.writeInt(value);
		}
		oos.flush();
	}

	public static void writeIntegerSetMap(ObjectOutputStream oos, SetMap<Integer, Integer> m) throws Exception {
		oos.writeInt(m.size());
		for (int key : m.keySet()) {
			oos.writeInt(key);
			writeIntegerSet(oos, m.get(key));
		}
		oos.flush();
	}

	public static void writeIntegerSetMap(String fileName, SetMap<Integer, Integer> m) throws Exception {
		ObjectOutputStream oos = openObjectOutputStream(fileName, false);
		writeIntegerSetMap(oos, m);
		oos.close();
	}

	public static void writeIntegerStringBidMap(ObjectOutputStream oos, BidMap<Integer, String> map) throws Exception {
		oos.writeInt(map.size());
		for (Entry<Integer, String> e : map.getKeyToValue().entrySet()) {
			oos.writeInt(e.getKey());
			oos.writeUTF(e.getValue());
		}
		oos.flush();
	}

	public static void writeIntegerStringMap(ObjectOutputStream oos, Map<Integer, String> m) throws Exception {
		oos.writeInt(m.size());
		for (Integer key : m.keySet()) {
			oos.writeInt(key);
			oos.writeUTF(m.get(key));
		}
		oos.flush();
	}

	public static void writeLongArray(ObjectOutputStream oos, long[] x) throws Exception {
		oos.writeInt(x.length);

		for (int i = 0; i < x.length; i++) {
			oos.writeLong(x[i]);
		}
	}

	public static void writeLongArray(String fileName, long[] x) throws Exception {
		ObjectOutputStream oos = openObjectOutputStream(fileName);
		writeLongArray(oos, x);
		oos.close();
	}

	public static void writeStringArray(ObjectOutputStream oos, String[] a) throws Exception {
		oos.writeInt(a.length);
		for (int i = 0; i < a.length; i++) {
			oos.writeUTF(a[i]);
		}
		oos.flush();
	}

	public static void writeStringCollection(ObjectOutputStream oos, Collection<String> c) throws Exception {
		oos.writeInt(c.size());
		Iterator<String> iter = c.iterator();
		while (iter.hasNext()) {
			oos.writeUTF(iter.next());
		}
		oos.flush();
	}

	public static void writeStringCollection(String fileName, Collection<String> c) throws Exception {
		System.out.printf("write [%d] at [%s].\n", c.size(), fileName);

		BufferedWriter bw = openBufferedWriter(fileName);
		bw.write(String.format("%s\t%d", LINE_SIZE, c.size()));
		for (String s : c) {
			bw.write(String.format("\n%s", s));
		}
		bw.flush();
		bw.close();
	}

	public static void writeStringCounter(ObjectOutputStream oos, Counter<String> x) throws Exception {
		oos.writeInt(x.size());
		for (String key : x.keySet()) {
			oos.writeUTF(key);
			oos.writeDouble(x.getCount(key));
		}
		oos.flush();
	}

	public static void writeStringCounterAsText(String fileName, Counter<String> c) throws Exception {
		writeStringCounterAsText(fileName, c, false);
	}

	public static void writeStringCounterAsText(String fileName, Counter<String> c, boolean alphabet_order) throws Exception {
		Timer timer = Timer.newTimer();

		BufferedWriter bw = openBufferedWriter(fileName, UTF_8, false);
		List<String> keys = new ArrayList<String>();
		if (alphabet_order) {
			keys = new ArrayList<String>(c.keySet());
			Collections.sort(keys);
		} else {
			keys = c.getSortedKeys();
		}
		bw.write(String.format("%s\t%d", LINE_SIZE, c.size()));
		for (int i = 0; i < keys.size(); i++) {
			String key = keys.get(i);
			double val = c.getCount(key);
			int tmp = (int) val;
			String output = null;
			if (val - tmp == 0) {
				output = String.format("\n%s\t%d", key, tmp);
			} else {
				output = String.format("\n%s\t%f", key, val);
			}
			bw.write(output);
		}
		bw.close();

		System.out.printf("write [%d] at [%s] - [%s]\n", c.size(), fileName, timer.stop());
	}

	public static void writeStringCounterMap(ObjectOutputStream oos, CounterMap<String, String> cm) throws Exception {
		oos.writeInt(cm.keySet().size());
		for (String key : cm.keySet()) {
			oos.writeUTF(key);
			writeStringCounter(oos, cm.getCounter(key));
		}
		oos.flush();
	}

	public static void writeStringCounterMap(String fileName, CounterMap<String, String> cm) throws Exception {
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeStringCounterMap(oos, cm);
		oos.close();
	}

	public static void writeStringCounterMapAsText(String fileName, CounterMap<String, String> cm) throws Exception {
		writeStringCounterMapAsText(fileName, cm, false);
	}

	public static void writeStringCounterMapAsText(String fileName, CounterMap<String, String> cm, boolean alphabet_order)
			throws Exception {
		Timer timer = Timer.newTimer();

		BufferedWriter bw = openBufferedWriter(fileName, UTF_8, false);
		List<String> keys = Generics.newArrayList();

		if (alphabet_order) {
			keys.addAll(cm.keySet());
			Collections.sort(keys);
		} else {
			keys = cm.getOutKeyCountSums().getSortedKeys();
		}

		bw.write(String.format("%s\t%d", LINE_SIZE, keys.size()));

		int num_entries = 0;
		double dv = 0;

		for (int i = 0; i < keys.size(); i++) {
			String outKey = keys.get(i);
			Counter<String> ic = cm.getCounter(outKey);
			dv = ic.totalCount();
			int tmp = (int) dv;

			if (dv - tmp == 0) {
				bw.write(String.format("\n%s\t%d", outKey, tmp));
			} else {
				bw.write(String.format("\n%s\t%f", outKey, dv));
			}

			for (String inKey : ic.getSortedKeys()) {
				dv = ic.getCount(inKey);
				tmp = (int) dv;

				if (dv - tmp == 0) {
					bw.write(String.format("\t%s:%d", inKey, tmp));
				} else {
					bw.write(String.format("\t%s:%f", inKey, dv));
				}

			}
			num_entries += ic.size();
		}
		bw.flush();
		bw.close();

		System.out.printf("write [%d] ents at [%s] - [%s]\n", num_entries, fileName, timer.stop());
	}

	public static void writeStringIndexer(ObjectOutputStream oos, Indexer<String> indexer) throws Exception {
		oos.writeInt(indexer.size());
		for (int i = 0; i < indexer.size(); i++) {
			oos.writeUTF(indexer.getObject(i));
		}
		oos.flush();
	}

	public static void writeStringIndexerAsText(BufferedWriter bw, Indexer<String> indexer) throws Exception {
		bw.write(String.format("%s\t%d", LINE_SIZE, indexer.size()));

		for (int i = 0; i < indexer.size(); i++) {
			bw.write(String.format("\n%s", indexer.getObject(i)));
		}
		bw.write("\n");

	}

	public static void writeStringIndexerAsText(String fileName, Indexer<String> indexer) throws Exception {
		TextFileWriter writer = new TextFileWriter(fileName);
		for (int i = 0; i < indexer.getObjects().size(); i++) {
			String label = indexer.getObject(i);
			writer.write(label + "\n");
		}
		writer.close();
	}

	public static void writeStringMapAsText(String fileName, Map<String, String> m) throws Exception {
		Timer timer = Timer.newTimer();

		BufferedWriter bw = openBufferedWriter(fileName, UTF_8, false);

		List<String> keys = new ArrayList<String>(m.keySet());
		Collections.sort(keys);

		bw.write(String.format("%s\t%d", LINE_SIZE, m.size()));
		for (int i = 0; i < keys.size(); i++) {
			String key = keys.get(i);
			String val = m.get(key);
			String output = String.format("\n%s\t%s", key, val);
			bw.write(output);
		}
		bw.close();

		System.out.printf("write [%d] ents at [%s] - [%s]\n", m.size(), fileName, timer.stop());
	}

	public static void writeStringSetMapAsText(String fileName, SetMap<String, String> sm) throws Exception {
		Timer timer = Timer.newTimer();

		BufferedWriter bw = openBufferedWriter(fileName, UTF_8, false);
		List<String> outKeys = Generics.newArrayList(sm.keySet());
		Collections.sort(outKeys);

		bw.write(String.format("%s\t%d", LINE_SIZE, outKeys.size()));

		int num_entries = 0;

		for (int i = 0; i < outKeys.size(); i++) {
			String outKey = outKeys.get(i);
			List<String> inKeys = Generics.newArrayList(sm.get(outKey));
			Collections.sort(inKeys);
			bw.write(String.format("\n%s\t%d", outKey, inKeys.size()));
			for (String inKey : inKeys) {
				bw.write(String.format("\t%s", inKey));
			}
			num_entries += inKeys.size();
		}
		bw.close();

		System.out.printf("write [%d] ents at [%s] - [%s]\n", num_entries, fileName, timer.stop());
	}

}
