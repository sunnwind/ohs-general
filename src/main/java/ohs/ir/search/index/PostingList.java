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
//		if (b.size() > 1) {
			ArrayMath.add(b.values(), -1, b.values());
//		}
	}

	public static ByteArray encode(IntegerArray a) throws Exception {
//		if (a.size() > 1) {
			ArrayMath.add(a.values(), 1, a.values());
//		}
		IndexCompression.gapEncode(a.values());
		
		for(int v : a) {
			if(v < 0) {
				System.err.println();
			}
		}
		
		return new ByteArray(IndexCompression.gammaEncodedOutputStream(a.values(), a.size()).toByteArray());
	}

	public static PostingList readPostingList(FileChannel fc) throws Exception {
		PostingList ret = null;
		if (fc.position() < fc.size()) {
			ret = toPostingList(FileUtils.readByteArrayMatrix(fc));
		}
		return ret;
	}

	public static void checkEncoding(IntegerArray a) throws Exception {
		ByteArray bs = encode(a.clone());
		IntegerArray b = new IntegerArray(new int[a.size()]);
		decode(bs, b);

		if (!a.equals(b)) {
			System.out.println();
		}
	}

	public static ByteArrayMatrix toByteArrayMatrix(PostingList pl) throws Exception {
		int w = pl.getWord();
		IntegerArray dseqs = pl.getDocSeqs().clone();
		IntegerArrayMatrix posData = pl.getPosData();

		checkEncoding(dseqs);

		ByteArrayMatrix ret = new ByteArrayMatrix(pl.getPosData().size() * 2 + 3);
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
		long[] ret = FileUtils.write(toByteArrayMatrix(pl), fc);

		boolean check = true;

		if (check) {
//			fc.position(ret[0]);
//			PostingList pl2 = readPostingList(fc);
//
//			pl.getDocSeqs().trimToSize();
//			pl.getPosData().trimToSize();
//
//			if (!pl.equals(pl2)) {
//				System.out.println();
//			}
			
			IntegerArray dseqs = pl.getDocSeqs();
			
			for(int i = 1 ;i < dseqs.size();i++) {
				int idx1 = dseqs.get(i-1);
				int idx2 = dseqs.get(i);
				
				if(idx1 >= idx2) {
					System.out.println();
				}
			}
			
		}

		return ret;
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

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		PostingList other = (PostingList) obj;
		if (dseqs == null) {
			if (other.dseqs != null)
				return false;
		} else if (!dseqs.equals(other.dseqs))
			return false;
		if (posData == null) {
			if (other.posData != null)
				return false;
		} else if (!posData.equals(other.posData))
			return false;
		if (w != other.w)
			return false;
		return true;
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

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((dseqs == null) ? 0 : dseqs.hashCode());
		result = prime * result + ((posData == null) ? 0 : posData.hashCode());
		result = prime * result + w;
		return result;
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
