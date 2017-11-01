package ohs.corpus.type;

import java.io.File;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.stanford.nlp.math.ArrayMath;
import ohs.fake.FNPath;
import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.nlp.ling.types.MDocument;
import ohs.types.number.IntegerArray;
import ohs.types.number.LongArray;
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
public class RawDocumentCollectionCreator {

	public static void create(List<String> inDirNames, RawDocumentCollectionCreator rdcc) throws Exception {
		Timer timer = Timer.newTimer();

		int cnt = 0;
		for (int i = 0; i < inDirNames.size(); i++) {
			String inDirName = inDirNames.get(i);
			RawDocumentCollection rdc = new RawDocumentCollection(inDirName);

			System.out.println(rdc.toString());

			rdcc.addAttrs(rdc.getAttrData().get(0));

			for (int j = 0; j < rdc.size(); j++) {
				ArrayList<String> vals = rdc.get(j);
				rdcc.addValues(vals);

				if (++cnt % 100000 == 0) {
					System.out.printf("[%d, %s, %s]\n", cnt, timer.stop(), inDirName);
				}
			}
			rdc.close();
			rdcc.checkCollectionSize();

			System.out.printf("[%d, %s, %s]\n", cnt, timer.stop(), inDirName);
		}
		rdcc.close();
		System.out.printf("[%d, %s]\n", cnt, timer.stop());
	}

	public static void createFromTokenizedData(String inDir, RawDocumentCollectionCreator rdcc) throws Exception {
		Timer timer = Timer.newTimer();
		int doc_cnt = 0;

		for (File file : FileUtils.getFilesUnder(inDir)) {
			for (String line : FileUtils.readLinesFromText(file)) {
				String[] ps = line.split("\t");

				ps = StrUtils.unwrap(ps);
				for (int i = 0; i < ps.length; i++) {
					ps[i] = ps[i].replace(StrUtils.LINE_REP, "\n");
					ps[i] = ps[i].replace(StrUtils.TAB_REP, "\t");
				}
				rdcc.addValues(ps);
			}
		}
		rdcc.checkCollectionSize();
		rdcc.close();
		System.out.printf("[%d, %s]\n", doc_cnt, timer.stop());
	}

	public static void createFromNaverNews(String inDir, RawDocumentCollectionCreator rdcc) throws Exception {
		Timer timer = Timer.newTimer();
		int doc_cnt = 0;

		for (File file : FileUtils.getFilesUnder(inDir)) {
			if (!file.getName().startsWith("2017")) {
				continue;
			}

			for (String line : FileUtils.readLinesFromText(file)) {
				String[] ps = StrUtils.unwrap(line.split("\t"));

				for (int i = 0; i < ps.length; i++) {

					if (i == 4) {
						MDocument d = MDocument.newDocument(ps[i]);

						List<String> l = Generics.newArrayList(d.size());

						for (int j = 0; j < d.size(); j++) {
							l.add(StrUtils.join(" ", d.get(j).getTokenStrings(0)));
						}

						String s = StrUtils.join("\n", l);

						ps[i] = s;
					}

					ps[i] = ps[i].replace(StrUtils.LINE_REP, "\n");
					ps[i] = ps[i].replace(StrUtils.TAB_REP, "\t");

				}
				rdcc.addValues(ps);
			}
		}
		rdcc.checkCollectionSize();
		rdcc.close();
		System.out.printf("[%d, %s]\n", doc_cnt, timer.stop());
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		boolean encode = false;

		// {
		// String[] attrs = { "sequential identifier", "MEDLINE identifier",
		// "Human-assigned MeSH terms (MH)", "Title (TI)",
		// "Publication type (PT)", "Abstract (AB)", "Author (AU)", "Source (SO)" };
		// String inDir = MIRPath.OHSUMED_COL_TOK_DIR;
		// String outDir = MIRPath.OHSUMED_COL_DC_DIR;
		// boolean append = false;
		//
		// RawDocumentCollectionCreator rdcc = new RawDocumentCollectionCreator(outDir,
		// append);
		// rdcc.setEncode(encode);
		// rdcc.addAttrs(Generics.newArrayList(attrs));
		// createFromTokenizedData(inDir, rdcc);
		//
		// // System.exit(0);
		// }
		//
		// {
		// String[] attrs = { "docid", "content" };
		// String inDir = MIRPath.TREC_GENO_2007_COL_TOK_DIR;
		// String outDir = MIRPath.TREC_GENO_2007_COL_DC_DIR;
		// boolean append = false;
		//
		// RawDocumentCollectionCreator rdcc = new RawDocumentCollectionCreator(outDir,
		// append);
		// rdcc.setEncode(encode);
		// rdcc.addAttrs(Generics.newArrayList(attrs));
		// createFromTokenizedData(inDir, rdcc);
		// }
		//
		// {
		// String[] attrs = { "pmcid", "title", "abs", "body", "kwds", "journal",
		// "pmid", "doi" };
		// String inDir = MIRPath.TREC_CDS_2016_COL_TOK_DIR;
		// String outDir = MIRPath.TREC_CDS_2016_COL_DC_DIR;
		// boolean append = false;
		// RawDocumentCollectionCreator rdcc = new RawDocumentCollectionCreator(outDir,
		// append);
		// rdcc.setEncode(encode);
		// rdcc.addAttrs(Generics.newArrayList(attrs));
		// createFromTokenizedData(inDir, rdcc);
		// }
		//
		// {
		// String[] attrs = { "pmcid", "journal", "title", "abs", "meshes" };
		// String inDir = MIRPath.TREC_PM_2017_COL_MEDLINE_TOK_DIR;
		// String outDir = MIRPath.TREC_PM_2017_COL_MEDLINE_DC_DIR;
		// boolean append = false;
		// RawDocumentCollectionCreator rdcc = new RawDocumentCollectionCreator(outDir,
		// append);
		// rdcc.setEncode(encode);
		// rdcc.addAttrs(Generics.newArrayList(attrs));
		// createFromTokenizedData(inDir, rdcc);
		// }
		//
		// {
		// String[] attrs = { "nctid", "brief_title", "official_title", "content",
		// "kwds", "meshes" };
		// String inDir = MIRPath.TREC_PM_2017_COL_CLINICAL_TOK_DIR;
		// String outDir = MIRPath.TREC_PM_2017_COL_CLINICAL_DC_DIR;
		// boolean append = false;
		// RawDocumentCollectionCreator rdcc = new RawDocumentCollectionCreator(outDir,
		// append);
		// rdcc.setEncode(encode);
		// rdcc.addAttrs(Generics.newArrayList(attrs));
		// createFromTokenizedData(inDir, rdcc);
		// }
		//
		// {
		// String[] attrs = { "pmcid", "title", "abs", "body", "kwds" };
		// String inDir = MIRPath.TREC_CDS_2014_COL_TOK_DIR;
		// String outDir = MIRPath.TREC_CDS_2014_COL_DC_DIR;
		// boolean append = false;
		// RawDocumentCollectionCreator rdcc = new RawDocumentCollectionCreator(outDir,
		// append);
		// rdcc.setEncode(encode);
		// rdcc.addAttrs(Generics.newArrayList(attrs));
		// createFromTokenizedData(inDir, rdcc);
		// }
		//
		// {
		// String[] attrs = { "id", "url", "title", "content", "phrs" };
		// String inDir = MIRPath.WIKI_COL_TOK_DIR;
		// String outDir = MIRPath.WIKI_COL_DC_DIR;
		// boolean append = false;
		// RawDocumentCollectionCreator rdcc = new RawDocumentCollectionCreator(outDir,
		// append);
		// rdcc.setEncode(encode);
		// rdcc.addAttrs(Generics.newArrayList(attrs));
		// createFromTokenizedData(inDir, rdcc);
		// }
		//
		// {
		// String[] attrs = { "pmid", "journal", "year", "mesh", "title", "abs" };
		// String inDir = MIRPath.BIOASQ_COL_TOK_DIR;
		// String outDir = MIRPath.BIOASQ_COL_DC_DIR;
		// boolean append = false;
		// RawDocumentCollectionCreator rdcc = new RawDocumentCollectionCreator(outDir,
		// append);
		// rdcc.setEncode(encode);
		// rdcc.addAttrs(Generics.newArrayList(attrs));
		// createFromTokenizedData(inDir, rdcc);
		// }
		//
		// {
		// String[] attrs = { "id", "content", "uri" };
		// String inDir = MIRPath.CLUEWEB_COL_TOK_DIR;
		// String outDir = MIRPath.CLUEWEB_COL_DC_DIR;
		// boolean append = false;
		// RawDocumentCollectionCreator rdcc = new RawDocumentCollectionCreator(outDir,
		// append);
		// rdcc.setEncode(encode);
		// rdcc.addAttrs(Generics.newArrayList(attrs));
		// createFromTokenizedData(inDir, rdcc);
		// }
		//
		// {
		// String[] attrs = { "uid", "date", "url", "content" };
		// String inDir = MIRPath.CLEF_EH_2014_COL_TOK_DIR;
		// String outDir = MIRPath.CLEF_EH_2014_COL_DC_DIR;
		// boolean append = false;
		// RawDocumentCollectionCreator rdcc = new RawDocumentCollectionCreator(outDir,
		// append);
		// rdcc.setEncode(encode);
		// rdcc.addAttrs(Generics.newArrayList(attrs));
		// createFromTokenizedData(inDir, rdcc);
		// }
		//
		// {
		// String[] attrs = { "docid", "title", "abs", "kwds" };
		// String inDir = "../../data/medical_ir/scopus/col/tok/";
		// String outDir = "../../data/medical_ir/scopus/col/dc/";
		// boolean append = false;
		// RawDocumentCollectionCreator rdcc = new RawDocumentCollectionCreator(outDir,
		// append);
		// rdcc.setEncode(encode);
		// rdcc.addAttrs(Generics.newArrayList(attrs));
		// createFromTokenizedData(inDir, rdcc);
		// }

		// {
		// String[] attrs = { "id", "oid", "cat1", "date", "title", "body", "url" };
		// String inDir = FNPath.NAVER_NEWS_COL_LINE_POS_DIR;
		// String outDir = FNPath.NAVER_NEWS_COL_DC_DIR;
		// boolean append = false;
		// RawDocumentCollectionCreator rdcc = new RawDocumentCollectionCreator(outDir,
		// append);
		// rdcc.setEncode(encode);
		// rdcc.addAttrs(Generics.newArrayList(attrs));
		// createFromTokenizedData(inDir, rdcc);
		// }

		{
			String[] attrs = { "id", "oid", "cat", "date", "content", "url" };
			String inDir = FNPath.NAVER_NEWS_COL_LINE_POS_DIR;
			String outDir = FNPath.NAVER_NEWS_COL_DC_DIR;
			boolean append = false;
			RawDocumentCollectionCreator rdcc = new RawDocumentCollectionCreator(outDir, append);
			rdcc.setEncode(encode);
			rdcc.addAttrs(Generics.newArrayList(attrs));
			createFromNaverNews(inDir, rdcc);
		}

		// {
		// String[] attrs = { "type", "cn", "kor_kwds", "eng_kwds", "kor_title",
		// "eng_title", "kor_abs", "eng_abs",
		// "kor_pos_kwds", "kor_pos_title", "kor_pos_abs" };
		// String inDir = KPPath.COL_LINE_POS_DIR;
		// String outDir = KPPath.COL_DC_DIR;
		// boolean append = false;
		// RawDocumentCollectionCreator rdcc = new RawDocumentCollectionCreator(outDir,
		// append);
		// rdcc.setEncode(encode);
		// rdcc.addAttrs(Generics.newArrayList(attrs));
		//
		// Timer timer = Timer.newTimer();
		// int doc_cnt = 0;
		//
		// for (File dir : new File(inDir).listFiles()) {
		// for (File file : FileUtils.getFilesUnder(dir)) {
		// for (String line : FileUtils.readLinesFromText(file)) {
		// String[] vals = StrUtils.unwrap(line.split("\t"));
		// for (int i = 0; i < vals.length; i++) {
		// vals[i] = vals[i].replace(StrUtils.LINE_REP, "\n");
		// vals[i] = vals[i].replace(StrUtils.TAB_REP, "\t");
		// }
		// rdcc.addValues(vals);
		// }
		// }
		// }
		//
		// rdcc.checkCollectionSize();
		// rdcc.close();
		// }

		// {
		// List<String> inDirNames = Generics.newArrayList();
		// inDirNames.add(MIRPath.OHSUMED_COL_DC_DIR);
		// inDirNames.add(MIRPath.TREC_PM_2017_COL_MEDLINE_DC_DIR);
		// inDirNames.add(MIRPath.TREC_PM_2017_COL_CLINICAL_DC_DIR);
		// inDirNames.add(MIRPath.TREC_GENO_2007_COL_DC_DIR);
		// inDirNames.add(MIRPath.BIOASQ_COL_DC_DIR);
		// inDirNames.add(MIRPath.TREC_CDS_2016_COL_DC_DIR);
		// inDirNames.add(MIRPath.WIKI_COL_DC_DIR);
		// String outDirName = MIRPath.DATA_DIR + "merged/col/dc/";
		//
		// RawDocumentCollectionCreator rdcc = new
		// RawDocumentCollectionCreator(outDirName, false);
		// rdcc.setEncode(encode);
		//
		// create(inDirNames, rdcc);
		// }
		//
		// {
		// DocumentCollectionCreator.main(args);
		// }

		System.out.println("process ends.");
	}

	private LongArray starts;

	private IntegerArray lens;

	private IntegerArray sizes;

	private File docFile;

	private File metaFile;

	private File dataDir;

	private FileChannel out1;

	private List<List<String>> attrData;

	private List<List<Boolean>> flagData;

	private int type = 0;

	private boolean encode = false;

	private int batch_size = 200;

	private List<ByteArrayMatrix> docs = Generics.newArrayList();

	private int init_col_size = 1000000;

	public RawDocumentCollectionCreator(String dataDirName, boolean append) throws Exception {
		starts = new LongArray(init_col_size);
		lens = new IntegerArray(init_col_size);
		sizes = new IntegerArray();

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
			File dataFile = new File(dataDir, RawDocumentCollection.DATA_NAME);
			File metaFile = new File(dataDir, RawDocumentCollection.META_NAME);

			if (dataFile.exists()) {
				dataFile.delete();
			}

			if (metaFile.exists()) {
				metaFile.delete();
			}
		}

		out1 = FileUtils.openFileChannel(docFile, "rw");
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
				// System.err.printf("[%d/%d] values - [%s]\n", vals.size(), attrs.size(),
				// StrUtils.join(", ", vals));
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

	public void checkCollectionSize() throws Exception {
		writeDocs();

		int sum = ArrayMath.sum(sizes.values());
		int new_size = starts.size() - sum;
		sizes.add(new_size);
	}

	public void close() throws Exception {
		writeDocs();

		out1.close();

		{
			FileChannel out2 = FileUtils.openFileChannel(metaFile, "rw");
			{
				starts.trimToSize();
				lens.trimToSize();
				sizes.trimToSize();

				DataCompression.encodeGaps(starts);

				ByteArrayMatrix data = new ByteArrayMatrix(3);
				data.add(DataCompression.encode(starts));
				data.add(DataCompression.encode(lens));
				data.add(DataCompression.encode(sizes));

				long[] info = FileUtils.write(data, out2);
			}

			{
				ByteArrayMatrix data = new ByteArrayMatrix(attrData.size());
				for (int i = 0; i < attrData.size(); i++) {
					List<String> attrs = attrData.get(i);
					data.add(DataCompression.encode(attrs));
				}
				long[] info = FileUtils.write(data, out2);
			}

			{
				ByteArrayMatrix data = new ByteArrayMatrix(flagData.size());

				for (int i = 0; i < flagData.size(); i++) {
					boolean[] flags = new boolean[flagData.get(i).size()];
					int j = 0;
					for (Boolean flag : flagData.get(i)) {
						flags[j++] = flag;
					}

					ByteBufferWrapper buf = new ByteBufferWrapper();
					buf.write(flags);

					byte[] aa = buf.getByteBuffer().array();

					ByteArray ba = buf.getByteArray();
					data.add(ba);
				}
				long[] info = FileUtils.write(data, out2);
			}

			out2.close();
		}
	}

	public List<Boolean> getEncodeFlags() {
		return flagData.get(type);
	}

	public void setEncode(boolean encode) {
		this.encode = encode;
	}

	public void setInitialCollectionSize(int init_col_size) {
		this.init_col_size = init_col_size;
	}

	public void setMaxBufferSize(int max_buf_size) {
		this.batch_size = max_buf_size;
	}

	private void writeDocs() throws Exception {
		for (ByteArrayMatrix m : docs) {
			long[] info = FileUtils.write(m, out1);
			starts.add(info[0]);
			lens.add((int) info[1]);
		}
		docs.clear();
	}

}
