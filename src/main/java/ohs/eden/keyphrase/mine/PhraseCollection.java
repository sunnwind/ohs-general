package ohs.eden.keyphrase.mine;

import java.io.File;
import java.nio.channels.FileChannel;
import java.util.List;

import ohs.corpus.type.DataCompression;
import ohs.corpus.type.DocumentCollection;
import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.types.number.LongArray;
import ohs.utils.Generics;

public class PhraseCollection {

	public static final String NAME = "freq_phrs.ser";

	public static final String META_NAME = "freq_phrs_meta.ser";

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// Set<String> stopwords =
		// FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE);

		// String[] dirs = { MIRPath.OHSUMED_COL_DC_DIR,
		// MIRPath.TREC_CDS_2016_COL_DC_DIR, MIRPath.BIOASQ_COL_DC_DIR,
		// MIRPath.WIKI_COL_DIR
		// };

		PhraseCollection pc = new PhraseCollection(MIRPath.OHSUMED_COL_DC_DIR);
		DocumentCollection ldc = new DocumentCollection(MIRPath.OHSUMED_COL_DC_DIR);

		System.out.println(ldc.size());
		System.out.println(pc.size());

		for (int i = 0; i < pc.size(); i++) {
			SPostingList pl = pc.getPostingList(i);
			System.out.println(pl.toString());

			String phrs = pl.getPhrase();
			int len_phrs = phrs.split("_").length;

			IntegerArray dseqs = pl.getDocseqs();
			IntegerArrayMatrix posData = pl.getPosData();

//			for (int j = 0; j < pl.size(); j++) {
//				Posting p = pl.getPosting(j);
//
//				int dseq = p.getDocseq();
//				IntegerArray poss = p.getPoss();
//
//				List<String> words = ldc.getWords(dseq);
//				int window_size = 3;
//
//				for (int start : poss) {
//					int end = start + len_phrs;
//					int left = Math.max(0, start - window_size);
//					int right = Math.min(end + window_size, words.size());
//
//					List<String> context = Generics.newArrayList();
//
//					for (int k = left; k < right; k++) {
//						String word = words.get(k);
//						if (k == start) {
//							context.add("[");
//						} else if (k == end) {
//							context.add("]");
//						}
//						context.add(word);
//					}
//
//					System.out.println("=> " + StrUtils.join(" ", context));
//				}
//			}

//			System.out.println();

		}

		System.out.println("process ends.");
	}

	private File dataDir;

	private LongArray starts;

	private IntegerArray lens;

	private IntegerArray phrsLens;

	private FileChannel fc;

	public PhraseCollection(String dataDir) throws Exception {
		this.dataDir = new File(dataDir);
		{
			FileChannel fc = FileUtils.openFileChannel(new File(dataDir, PhraseCollection.META_NAME), "rw");
			ByteArrayMatrix data = FileUtils.readByteArrayMatrix(fc);
			fc.close();

			int i = 0;
			starts = DataCompression.decodeToLongArray(data.get(i++));
			lens = DataCompression.decodeToIntegerArray(data.get(i++));
			phrsLens = DataCompression.decodeToIntegerArray(data.get(i++));

			DataCompression.decodeGaps(starts);
		}

		fc = FileUtils.openFileChannel(new File(dataDir, PhraseCollection.NAME), "r");
	}

	public IntegerArray getPhraseLengths() {
		return phrsLens;
	}

	public SPostingList getPostingList(int i) throws Exception {
		SPostingList ret = null;

		if (i < 0 || i >= starts.size()) {
			return ret;
		}

		long start = starts.get(i);

		if (start < 0) {
			return ret;
		}

		fc.position(start);
		SPostingList pl = SPostingList.readPostingList(fc, true);
		return pl;
	}

	public List<SPostingList> getPostingLists(int i, int j) throws Exception {
		long start = starts.get(i);
		int size1 = 0;
		for (int k = i; k < j && k < size(); k++) {
			size1 += lens.get(k);
		}
		fc.position(start);

		ByteArray data = FileUtils.readByteArray(fc, size1);
		ByteBufferWrapper buf = new ByteBufferWrapper(data);

		List<SPostingList> ret = Generics.newArrayList(j - i);

		for (int k = i; k < j; k++) {
			SPostingList pl = SPostingList.toPostingList(buf.readByteArrayMatrix(), true);
			ret.add(pl);
		}
		return ret;
	}

	public int size() {
		return starts.size();
	}

}
