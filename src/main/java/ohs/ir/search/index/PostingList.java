package ohs.ir.search.index;

import java.nio.channels.FileChannel;
import java.util.List;

import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteArrayUtils;
import ohs.io.FileUtils;
import ohs.math.ArrayMath;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.ByteSize;
import ohs.utils.Generics;

public class PostingList {

	public static void decode(ByteArray a, IntegerArray b) throws Exception {
		IndexCompression.gammaDecode(a.values(), b.values());
		IndexCompression.gapDecode(b.values());
		ArrayMath.add(b.values(), -1, b.values());
	}

	public static ByteArray encode(IntegerArray a) throws Exception {
		ArrayMath.add(a.values(), 1, a.values());
		IndexCompression.gapEncode(a.values());
		return new ByteArray(IndexCompression.gammaEncodedOutputStream(a.values(), a.size()).toByteArray());
	}

	public static PostingList readPostingList(FileChannel fc) throws Exception {
		PostingList ret = null;
		if (fc.position() < fc.size()) {
			ret = toPostingList(FileUtils.readByteArrayMatrix(fc));
		}
		return ret;
	}

	public static List<PostingList> readPostingLists(FileChannel fc) throws Exception {
		List<PostingList> ret = Generics.newLinkedList();
		if (fc.position() < fc.size()) {
			ret.add(toPostingList(FileUtils.readByteArrayMatrix(fc)));
		}
		return ret;
	}

	public static int sizeOfByteBuffer(PostingList pl) {
		int ret = Integer.BYTES;
		ret += Integer.BYTES * pl.getDocSeqs().size() * 2;
		ret += Integer.BYTES * pl.getPosData().sizeOfEntries();
		ret += Integer.BYTES * pl.getPosData().size();
		return ret;
	}

	public static ByteArrayMatrix toByteArrayMatrix(PostingList pl) throws Exception {
		int w = pl.getWord();
		IntegerArray dseqs = pl.getDocSeqs().clone();
		IntegerArrayMatrix posData = pl.getPosData();

		ByteArrayMatrix ret = new ByteArrayMatrix(pl.getPosData().size() * 2 + 1);
		ret.add(ByteArrayUtils.toByteArray(w));
		ret.add(ByteArrayUtils.toByteArray(dseqs.size()));
		ret.add(encode(dseqs));

		for (IntegerArray poss : posData) {
			poss = poss.clone();
			ret.add(ByteArrayUtils.toByteArray(poss.size()));
			ret.add(encode(poss));
		}

		return new ByteArrayMatrix(ret);
	}

	public static PostingList toPostingList(ByteArrayMatrix data) throws Exception {
		int i = 0;
		int w = ByteArrayUtils.toInteger(data.get(i++));
		int pl_size = ByteArrayUtils.toInteger(data.get(i++));

		IntegerArray dseqs = new IntegerArray(new int[pl_size]);
		IntegerArrayMatrix posData = new IntegerArrayMatrix(pl_size);

		decode(data.get(i++), dseqs);

		for (int j = 0; j < dseqs.size(); j++) {
			int len = ByteArrayUtils.toInteger(data.get(i++));
			IntegerArray poss = new IntegerArray(new int[len]);
			decode(data.get(i++), poss);
			posData.add(poss);
		}

		return new PostingList(w, dseqs, posData);
	}

	public static long[] writePostingList(PostingList pl, FileChannel fc) throws Exception {
		return FileUtils.write(toByteArrayMatrix(pl), fc);
	}

	private int w;

	private IntegerArray dseqs;

	private IntegerArray cnts;

	private IntegerArrayMatrix posData;

	private IntegerArrayMatrix endPosData;

	private int total_cnt;

	public PostingList(int w, IntegerArray dseqs, IntegerArray cnts, IntegerArrayMatrix posData) {
		this.w = w;
		this.dseqs = dseqs;
		this.cnts = cnts;
		this.posData = posData;
	}

	public PostingList(int w, IntegerArray dseqs, IntegerArrayMatrix posData) {
		this.w = w;
		this.dseqs = dseqs;
		this.posData = posData;
		cnts = new IntegerArray(dseqs.size());
		for (IntegerArray poss : posData) {
			cnts.add(poss.size());
			total_cnt += poss.size();
		}
	}

	public int getCount() {
		return total_cnt;
	}

	public IntegerArray getCounts() {
		return cnts;
	}

	public IntegerArray getDocSeqs() {
		return dseqs;
	}

	public IntegerArrayMatrix getEndPosData() {
		return endPosData;
	}

	public IntegerArrayMatrix getPosData() {
		return posData;
	}

	public IntegerArray getPoss(int i) throws Exception {
		return posData.get(i);
	}

	public Posting getPosting(int i) {
		return new Posting(dseqs.get(i), posData.get(i), endPosData == null ? null : endPosData.get(i));
	}

	public int getWord() {
		return w;
	}

	public void setEndPosData(IntegerArrayMatrix starts) {
		this.endPosData = starts;
	}

	public void setPosData(IntegerArrayMatrix posData) {
		this.posData = posData;
	}

	public void setWord(int w) {
		this.w = w;
	}

	public int size() {
		return dseqs.size();
	}

	public String toString() {

		long bytes = ByteArrayUtils.sizeOfByteBuffer(dseqs) * 2 + ByteArrayUtils.sizeOfByteBuffer(posData);
		ByteSize bs = new ByteSize(bytes);
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("w=[%d], docs=[%d], cnt=[%d], mem=%s", w, size(), getCount(), bs));
		return sb.toString();
	}

}
