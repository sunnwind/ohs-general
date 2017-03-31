package ohs.eden.linker;

import java.io.BufferedWriter;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.string.search.ppss.Gram;
import ohs.string.search.ppss.GramGenerator;
import ohs.string.search.ppss.StringRecord;
import ohs.string.sim.EditDistance;
import ohs.string.sim.Jaccard;
import ohs.string.sim.Jaro;
import ohs.string.sim.Sequence;
import ohs.string.sim.SequenceFactory;
import ohs.string.sim.SmithWaterman;
import ohs.string.sim.StringScorer;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListMap;
import ohs.types.generic.MapMap;
import ohs.utils.Generics;
import ohs.utils.Generics.ListType;
import ohs.utils.Generics.MapType;
import ohs.utils.Timer;

/**
 * 
 * @author Heung-Seon Oh
 */
public class StringSearcher implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8740333778747553831L;

	private Map<Integer, StringRecord> srs;

	private ListMap<Integer, Integer> index;

	private Indexer<String> gramIndexer;

	private GramGenerator gg;

	private MapMap<String, Integer, List<Double>> cache;

	private int top_k = 100;

	private int q = 3;

	private int tau = 3;

	private int prefix_size = q * tau + 1;

	private int[] gram_dfs;

	private int[] gram_cnts;

	private List<StringScorer<Character>> strScorers;

	private ListMap<Integer, Double> simScores;

	private Counter<Integer> cnds;

	private boolean makeLog = false;

	private StringBuffer logBuff;

	public StringSearcher() {
		this(3);
	}

	public StringSearcher(int q) {
		this.q = q;

		gg = new GramGenerator(q);
		cache = new MapMap<String, Integer, List<Double>>(1000, MapType.WEAK_HASH_MAP, MapType.WEAK_HASH_MAP);

		strScorers = Generics.newArrayList();

		strScorers.add(new EditDistance<Character>());
		strScorers.add(new SmithWaterman<Character>());
		strScorers.add(new Jaro<Character>());
		strScorers.add(new Jaccard<Character>());
		simScores = Generics.newListMap();

	}

	private void buildGramIndexer(List<StringRecord> input) {
		Timer timer = new Timer();
		timer.start();

		int chunk_size = input.size() / 100;

		Counter<Integer> gramDFs = Generics.newCounter();
		Counter<Integer> gramCnts = Generics.newCounter();

		if (gram_dfs.length > 0) {
			for (int i = 0; i < gram_dfs.length; i++) {
				gramDFs.incrementCount(i, gram_dfs[i]);
				gramCnts.incrementCount(i, gram_cnts[i]);
			}
		}

		for (int i = 0; i < input.size(); i++) {
			if ((i + 1) % chunk_size == 0) {
				int progess = (int) ((i + 1f) / input.size() * 100);
				System.out.printf("\r[%d percent, %s]", progess, timer.stop());
			}

			StringRecord sr = input.get(i);
			Gram[] grams = gg.generateQGrams(sr.getString().toLowerCase());

			if (grams.length == 0) {
				continue;
			}

			Set<Integer> gids = Generics.newHashSet();
			for (int j = 0; j < grams.length; j++) {
				int gid = gramIndexer.getIndex(grams[j].getString());
				gids.add(gid);
				gramCnts.incrementCount(gid, 1);
			}

			for (int gid : gids) {
				gramDFs.incrementCount(gid, 1);
			}
		}
		System.out.printf("\r[%d percent, %s]\n", 100, timer.stop());
		System.out.printf("built gram indexer [%d, %s].\n", gramIndexer.size(), timer.stop());

		gram_dfs = new int[gramIndexer.size()];
		gram_cnts = new int[gramIndexer.size()];

		for (Entry<Integer, Double> e : gramDFs.entrySet()) {
			int gid = e.getKey();
			gram_dfs[gid] = e.getValue().intValue();
			gram_cnts[gid] = (int) gramCnts.getCount(gid);
		}
	}

	public int[] getGramCounts() {
		return gram_cnts;
	}

	public int[] getGramDocumentFreqs() {
		return gram_dfs;
	}

	public GramGenerator getGramGenerator() {
		return gg;
	}

	public Indexer<String> getGramIndexer() {
		return gramIndexer;
	}

	public StringBuffer getLogBuffer() {
		return logBuff;
	}

	public List<StringScorer<Character>> getSimScorers() {
		return strScorers;
	}

	public ListMap<Integer, Double> getSimScores() {
		return simScores;
	}

	public Map<Integer, StringRecord> getStringRecords() {
		return srs;
	}

	public int getTopK() {
		return top_k;
	}

	public void index(List<StringRecord> input, boolean append) {
		System.out.printf("index [%s] records.\n", input.size());

		if (index == null && !append) {
			gramIndexer = Generics.newIndexer();
			index = Generics.newListMap(1000, MapType.HASH_MAP, ListType.ARRAY_LIST);
			srs = Generics.newHashMap();
			gram_dfs = new int[0];
			gram_cnts = new int[0];
		}

		buildGramIndexer(input);
		int num_records = srs.size() + input.size();

		Timer timer = new Timer();
		timer.start();

		int chunk_size = input.size() / 100;

		for (int i = 0; i < input.size(); i++) {

			if ((i + 1) % chunk_size == 0) {
				int progess = (int) ((i + 1f) / input.size() * 100);
				System.out.printf("\r[%d percent, %s]", progess, timer.stop());
			}

			StringRecord sr = input.get(i);
			Gram[] grams = gg.generateQGrams(sr.getString().toLowerCase());
			if (grams.length == 0) {
				continue;
			}

			Counter<Integer> gWeights = Generics.newCounter();

			for (int j = 0; j < grams.length; j++) {
				int gid = gramIndexer.indexOf(grams[j].getString());
				if (gid != -1 && !gWeights.containsKey(gid)) {
					int df = gram_dfs[gid];
					double idf = Math.log((num_records + 1.0) / df);
					gWeights.setCount(gid, idf);
				}
			}

			List<Integer> gids = gWeights.getSortedKeys();

			int cutoff = prefix_size;

			for (int j = 0; j < gids.size() && j < cutoff; j++) {
				index.put(gids.get(j), sr.getId());
			}

			srs.put(sr.getId(), sr);
		}

		System.out.printf("\r[%d percent, %s]\n", 100, timer.stop());

		for (int gid : index.keySet()) {
			Collections.sort(index.get(gid));
		}
	}

	public String info() {
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("string records:\t%d\n", srs.size()));

		{
			Counter<Integer> c = new Counter<Integer>();

			int max = -Integer.MAX_VALUE;
			int min = Integer.MAX_VALUE;
			double num_chars = 0;

			for (StringRecord sr : srs.values()) {
				c.incrementCount(sr.getString().length(), 1);
				max = Math.max(max, sr.getString().length());
				min = Math.min(min, sr.getString().length());
				num_chars += sr.getString().length();
			}
			double avg_chars = num_chars / srs.size();
			sb.append(String.format("max record length:\t%d\n", max));
			sb.append(String.format("min record length:\t%d\n", min));
			sb.append(String.format("avg record length:\t%f\n", avg_chars));
		}

		{
			int max = -Integer.MAX_VALUE;
			int min = Integer.MAX_VALUE;
			int num_records = 0;

			for (int qid : index.keySet()) {
				List<Integer> rids = index.get(qid, false);
				max = Math.max(max, rids.size());
				min = Math.min(min, rids.size());
				num_records += rids.size();
			}
			double avg_records = 1f * num_records / index.size();
			sb.append(String.format("q-grams:\t%d\n", index.size()));
			sb.append(String.format("max postings:\t%d\n", max));
			sb.append(String.format("min SPostingList:\t%d\n", min));
			sb.append(String.format("avg SPostingList:\t%f", avg_records));
		}
		return sb.toString();
	}

	public void read(ObjectInputStream ois) throws Exception {
		Timer timer = new Timer();
		timer.start();

		int size = ois.readInt();
		srs = Generics.newHashMap(size);
		for (int i = 0; i < size; i++) {
			StringRecord sr = new StringRecord();
			sr.read(ois);
			srs.put(sr.getId(), sr);
		}

		top_k = ois.readInt();
		q = ois.readInt();
		tau = ois.readInt();
		prefix_size = ois.readInt();

		gg = new GramGenerator(q);
		gramIndexer = FileUtils.readStringIndexer(ois);
		gram_dfs = FileUtils.readIntegers(ois);
		gram_cnts = FileUtils.readIntegers(ois);

		int size3 = ois.readInt();
		index = new ListMap<Integer, Integer>(size3, MapType.HASH_MAP, ListType.ARRAY_LIST);

		for (int i = 0; i < size3; i++) {
			index.put(ois.readInt(), FileUtils.readIntegerList(ois));
		}

		System.out.printf("read [%s] - [%s]\n", this.getClass().getName(), timer.stop());
	}

	public Counter<StringRecord> search(String s) {
		Gram[] grams = gg.generateQGrams(s.toLowerCase());

		if (grams.length == 0) {
			return Generics.newCounter();
		}

		Counter<Integer> gramIDFs = Generics.newCounter();

		for (int i = 0; i < grams.length; i++) {
			int gid = gramIndexer.indexOf(grams[i].getString());
			if (gid < 0) {
				continue;
			}
			int df = gram_dfs[gid];
			double idf = Math.log((srs.size() + 1.0) / df);
			gramIDFs.setCount(gid, idf);
		}

		cnds = Generics.newCounter();

		List<Integer> gids = gramIDFs.getSortedKeys();

		for (int i = 0; i < gids.size() && i < prefix_size; i++) {
			int gid = gids.get(i);
			List<Integer> rids = index.get(gid, false);
			if (rids != null) {
				double idf = gramIDFs.getCount(gid);
				for (int rid : rids) {
					cnds.incrementCount(rid, idf);
				}
			}
		}

		logBuff = new StringBuffer();

		if (makeLog) {
			logBuff.append(String.format("Input:\t%s", s));
		}

		Counter<StringRecord> ret = Generics.newCounter();
		simScores = Generics.newListMap(top_k);

		Sequence<Character> ss = SequenceFactory.newCharSequence(s.toLowerCase());

		List<Integer> rids = cnds.getSortedKeys();

		for (int i = 0; i < rids.size() && i < top_k; i++) {
			int rid = rids.get(i);
			StringRecord sr = srs.get(rid);
			double idf_sum = cnds.getCount(rid);

			Sequence<Character> tt = SequenceFactory.newCharSequence(sr.getString().toLowerCase());

			List<Double> scores = cache.get(s, sr.getId(), false);

			if (scores == null) {
				scores = Generics.newArrayList(strScorers.size());
				for (int j = 0; j < strScorers.size(); j++) {
					double sim = strScorers.get(j).getSimilarity(ss, tt);
					scores.add(sim);
				}
				cache.put(s, sr.getId(), scores);
			}

			simScores.put(sr.getId(), scores);

			double sim_score_sum = 0;
			for (int j = 0; j < scores.size(); j++) {
				sim_score_sum += scores.get(j).doubleValue();
			}
			double avg_sim_score = sim_score_sum / scores.size();
			double score = avg_sim_score;

			if (makeLog) {
				logBuff.append("\n" + sr.getId() + "\t" + sr.getString());
				for (int j = 0; j < scores.size(); j++) {
					logBuff.append(String.format("\t%f", scores.get(j)));
				}
				logBuff.append(String.format("\t%f", avg_sim_score));
				logBuff.append(String.format("\t%f", idf_sum));
				logBuff.append(String.format("\t%f", score));
			}

			ret.setCount(sr, score);
		}
		return ret;
	}

	public void setMakeLog(boolean makeLog) {
		this.makeLog = makeLog;
	}

	public void setTopK(int top_k) {
		this.top_k = top_k;
	}

	public void write(ObjectOutputStream oos) throws Exception {
		Timer timer = new Timer();
		timer.start();

		oos.writeInt(srs.size());
		for (StringRecord sr : srs.values()) {
			sr.write(oos);
		}

		oos.writeInt(top_k);
		oos.writeInt(q);
		oos.writeInt(tau);
		oos.writeInt(prefix_size);

		FileUtils.writeStringCollection(oos, gramIndexer.getObjects());
		FileUtils.writeIntegers(oos, gram_dfs);
		FileUtils.writeIntegers(oos, gram_cnts);

		oos.writeInt(index.size());
		Iterator<Integer> iter = index.keySet().iterator();

		while (iter.hasNext()) {
			int gid = iter.next();
			oos.writeInt(gid);
			FileUtils.writeIntegerCollection(oos, index.get(gid));
		}
		oos.flush();

		System.out.printf("write [%s] - [%s]\n", this.getClass().getName(), timer.stop());
	}

	public void write(String fileName) throws Exception {
		BufferedWriter writer = FileUtils.openBufferedWriter(fileName);
		writer.close();
	}
}
