package ohs.naver.cluster;

import java.io.File;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.List;

import ohs.corpus.type.DataCompression;
import ohs.corpus.type.RawDocumentCollection;
import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.types.number.IntegerArray;
import ohs.types.number.LongArray;
import ohs.types.number.ShortArray;
import ohs.utils.ByteSize;
import ohs.utils.ByteSize.Type;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

/**
 * 
 * 
 * http://stackoverflow.com/questions/5534070/effectively-compress-strings-of-10-1000-characters-in-java
 * 
 * @author ohs
 *
 */
public class NaverRawDocumentCollectionCreator {

	public static void createFromTokenizedData(NaverRawDocumentCollectionCreator rdc, String inDir) throws Exception {
		Timer timer = Timer.newTimer();
		int doc_cnt = 0;

		for (File file : FileUtils.getFilesUnder(inDir)) {
			// if (cnt++ > 0) {
			// break;
			// }
			for (String line : FileUtils.readLinesFromText(file)) {
				String[] vals = StrUtils.unwrap(line.split("\t"));
				for (int i = 0; i < vals.length; i++) {
					vals[i] = vals[i].replace(StrUtils.LINE_REP, "\n");
					vals[i] = vals[i].replace(StrUtils.TAB_REP, "\t");
				}
				rdc.addValues(vals);

				// if (++doc_cnt % 20000 == 0) {
				// System.out.printf("[%d, %s]\n", doc_cnt, timer.stop());
				// }
			}
		}
		System.out.printf("[%d, %s]\n", doc_cnt, timer.stop());
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// {
		// String[] attrs = { "title", "section", "body" };
		// String inDir = "../../data/naver_news/col/line/";
		// String outDir = "../../data/naver_news/col/dc/";
		// boolean append = false;
		// NaverRawDocumentCollectionCreator dcc = new NaverRawDocumentCollectionCreator(outDir, append);
		// dcc.addAttrs(Generics.newArrayList(attrs));
		// createFromTokenizedData(dcc, inDir);
		// dcc.close();
		// }

		{
			String[] attrs = { "title" };
			String inDir = "../../data/naver_news/col/line/";
			String outDir = "../../data/naver_news/col/dc/";
			boolean append = false;
			NaverRawDocumentCollectionCreator dcc = new NaverRawDocumentCollectionCreator(outDir, append);
			dcc.addAttrs(Generics.newArrayList(attrs));
			createFromTokenizedData(dcc, inDir);
			dcc.close();
		}

		System.out.println("process ends.");
	}

	private List<Long> starts;

	private List<Integer> lens;

	private List<Integer> types;

	private File docFile;

	private File metaFile;

	private File dataDir;

	private FileChannel fc;

	private List<List<String>> attrData;

	private List<List<Boolean>> flagData;

	private int type = 0;

	private boolean encode = false;

	private int batch_size = 200;

	private List<ByteArrayMatrix> docs = Generics.newLinkedList();

	private ByteBufferWrapper buf = new ByteBufferWrapper(new ByteSize(64, Type.MEGA).getBytes());

	public NaverRawDocumentCollectionCreator(String dataDirName, boolean append) throws Exception {
		starts = Generics.newLinkedList();
		types = Generics.newLinkedList();
		lens = Generics.newLinkedList();

		attrData = Generics.newArrayList();
		flagData = Generics.newArrayList();

		dataDir = new File(dataDirName);
		docFile = new File(dataDir, RawDocumentCollection.DATA_NAME);
		metaFile = new File(dataDir, RawDocumentCollection.META_NAME);

		if (append) {
			// FileChannel fc2 = FileUtils.openFileChannel(metaFile, "r");
			// {
			// ByteArrayMatrix data =
			// ByteArrayUtils.toByteArrayMatrix(FileUtils.readByteArray(fc2));
			// starts = DataCompression.decodeToLongArray(data.get(0));
			// types = DataCompression.decodeToShortArray(data.get(1));
			// }
			//
			// {
			// ByteArrayMatrix data =
			// ByteArrayUtils.toByteArrayMatrix(FileUtils.readByteArray(fc2));
			// attrData = Generics.newArrayList(data.size());
			// for (int i = 0; i < data.size(); i++) {
			// List<String> attrs = Generics.newArrayList();
			// for (String attr :
			// DataCompression.decodeToString(data.get(i)).split("\t")) {
			// attrs.add(attr);
			// }
			// attrData.add(attrs);
			// }
			// type = attrData.size();
			// }
		} else {
			FileUtils.deleteFilesUnder(dataDir);
		}

		fc = FileUtils.openFileChannel(docFile, "rw");
	}

	public void addAttrs(List<String> attrs) {
		type = attrData.size();
		attrData.add(attrs);

		List<Boolean> flags = Generics.newArrayList(attrs.size());
		for (String attr : attrs) {
			flags.add(encode);
		}
		flagData.add(flags);
	}

	public void addValues(List<String> vals) throws Exception {
		List<String> attrs = attrData.get(type);
		if (attrs.size() > 0) {
			if (attrs.size() != vals.size()) {
				// System.err.printf("[%d/%d] values - [%s]\n", vals.size(), attrs.size(), StrUtils.join(", ", vals));
				System.err.printf("[%d/%d] values\n", vals.size(), attrs.size());
				return;
			}
		}

		List<Boolean> flags = flagData.get(type);

		ByteArrayMatrix doc = new ByteArrayMatrix(vals.size());

		for (int i = 0; i < vals.size(); i++) {
			String val = vals.get(i);
			doc.add(flags.get(i) ? DataCompression.encode(val) : new ByteArray(val.getBytes()));
		}
		if (docs.size() == batch_size) {
			writeDocs();
		}
		docs.add(doc);
	}

	public void addValues(String[] vals) throws Exception {
		addValues(Arrays.asList(vals));
	}

	public void close() throws Exception {
		writeDocs();

		fc.close();

		FileChannel fc = FileUtils.openFileChannel(metaFile, "rw");

		{

			LongArray ss = new LongArray(starts);
			IntegerArray ls = new IntegerArray(lens);
			ShortArray ts = new ShortArray(starts.size());

			for (int type : types) {
				ts.add((short) type);
			}

			DataCompression.encodeGaps(ss);

			ByteArrayMatrix data = new ByteArrayMatrix(3);
			data.add(DataCompression.encode(ss));
			data.add(DataCompression.encode(ls));
			data.add(DataCompression.encode(ts));
			long[] info = FileUtils.write(data, buf, fc);
		}

		{
			ByteArrayMatrix data = new ByteArrayMatrix(attrData.size());
			for (int i = 0; i < attrData.size(); i++) {
				List<String> attrs = attrData.get(i);
				data.add(DataCompression.encode(attrs));
			}
			long[] info = FileUtils.write(data, buf, fc);
		}

		{
			byte[][] data = new byte[flagData.size()][];
			for (int i = 0; i < flagData.size(); i++) {
				boolean[] flags = new boolean[flagData.get(i).size()];
				int j = 0;
				for (Boolean flag : flagData.get(i)) {
					flags[j++] = flag;
				}

				ByteBufferWrapper buf = new ByteBufferWrapper();
				buf.write(flags);

				data[i] = buf.toByteArray().values();
			}
			long[] info = FileUtils.write(new ByteArrayMatrix(data), buf, fc);
		}

		fc.close();
	}

	public List<Boolean> getEncodeFlags() {
		return flagData.get(type);
	}

	public void setEncode(boolean encode) {
		this.encode = encode;
	}

	public void setMaxBufferSize(int max_buf_size) {
		this.batch_size = max_buf_size;
	}

	private void writeDocs() throws Exception {
		for (ByteArrayMatrix m : docs) {
			long[] info = FileUtils.write(m, buf, fc);
			starts.add(info[0]);
			lens.add((int) info[1]);
			types.add(type);
		}
		docs.clear();
	}

}
