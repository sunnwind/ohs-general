package ohs.ir.eval;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.NumberFormat;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import ohs.io.FileUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.utils.Generics;

public class Performance {

	private String name;

	private int top_n;

	private Counter<MetricType> scores;

	private Counter<MetricType> imprs;

	private Counter<MetricType> ris;

	private CounterMap<MetricType, String> queryScores;

	private CounterMap<MetricType, String> queryImprs;

	public Performance() {

	}
	

	public Performance(String fileName) throws Exception {
		readObject(fileName);
	}

	public Performance(int top_n, CounterMap<MetricType, String> queryScores) {
		this("", top_n, queryScores);
	}

	public Performance(String name, int top_n, CounterMap<MetricType, String> queryScores) {
		this.name = name;
		this.top_n = top_n;
		this.queryScores = queryScores;

		scores = Generics.newCounter();
		imprs = Generics.newCounter();
		queryImprs = Generics.newCounterMap();

		ris = Generics.newCounter();

		compute();
	}

	private void compute() {
		scores.setCount(MetricType.RETRIEVED, queryScores.getCounter(MetricType.RETRIEVED).totalCount());
		scores.setCount(MetricType.RELEVANT, queryScores.getCounter(MetricType.RELEVANT).totalCount());
		scores.setCount(MetricType.RELEVANT_IN_RET, queryScores.getCounter(MetricType.RELEVANT_IN_RET).totalCount());
		scores.setCount(MetricType.RELEVANT_AT, queryScores.getCounter(MetricType.RELEVANT_AT).totalCount());

		scores.setCount(MetricType.P, queryScores.getCounter(MetricType.P).average());
		scores.setCount(MetricType.MAP, queryScores.getCounter(MetricType.AP).average());
		scores.setCount(MetricType.NDCG, queryScores.getCounter(MetricType.NDCG).average());
	}

	public double get(MetricType mt) {
		return scores.getCount(mt);
	}

	public Counter<MetricType> getImprovements() {
		return imprs;
	}

	public String getName() {
		return name;
	}

	public CounterMap<MetricType, String> getQueryImprovements() {
		return queryImprs;
	}

	public CounterMap<MetricType, String> getQueryScores() {
		return queryScores;
	}

	public Counter<MetricType> getRobustnessIndexes() {
		return ris;
	}

	public Counter<MetricType> getScores() {
		return scores;
	}

	public int getTopN() {
		return top_n;
	}

	private Counter<MetricType> readCounter(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		Counter<MetricType> ret = Generics.newCounter(size);
		for (int i = 0; i < size; i++) {
			ret.setCount(MetricType.values()[ois.readInt()], ois.readDouble());
		}
		return ret;
	}

	private CounterMap<MetricType, String> readCounterMap(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		CounterMap<MetricType, String> ret = Generics.newCounterMap(size);
		for (int i = 0; i < size; i++) {
			ret.setCounter(MetricType.values()[ois.readInt()], FileUtils.readStringCounter(ois));
		}
		return ret;
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		top_n = ois.readInt();
		name = ois.readUTF();
		scores = readCounter(ois);
		imprs = readCounter(ois);
		ris = readCounter(ois);
		queryScores = readCounterMap(ois);
		queryImprs = readCounterMap(ois);
	}

	public void readObject(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	@Override
	public String toString() {
		return toString(false);
	}

	private void append(StringBuffer sb, double value) {
		int tmp = (int) value;

		if (value - tmp == 0) {
			sb.append(String.format("\t%d", tmp));
		} else {
			sb.append(String.format("\t%f", value));
		}
	}

	public String toString(boolean show_details) {
		StringBuffer sb = new StringBuffer();

		NumberFormat nf = NumberFormat.getInstance();
		nf.setMinimumFractionDigits(5);
		nf.setGroupingUsed(false);

		Set<String> qids = new TreeSet<String>(queryScores.getCounter(MetricType.RETRIEVED).keySet());

		sb.append(String.format("<Performance for Top-%d>", top_n));
		sb.append(String.format("\nQueries:\t%d", qids.size()));
		sb.append("\nMetric\tScore\tImpr\tRI");

		MetricType[] mts = MetricType.values();

		for (MetricType mt : mts) {
			if (scores.containsKey(mt)) {

				sb.append("\n" + mt.toString());
				append(sb, scores.getCount(mt));
				append(sb, imprs.getCount(mt));
				append(sb, ris.getCount(mt));
			}
		}

		if (show_details) {

			Set<MetricType> toShowImprs = Generics.newHashSet();
			toShowImprs.add(MetricType.P);
			toShowImprs.add(MetricType.AP);
			toShowImprs.add(MetricType.NDCG);

			List<MetricType> toShow = Generics.newArrayList();

			for (MetricType mt : mts) {
				if (queryScores.containsKey(mt) && queryScores.getCounter(mt).size() > 0) {
					toShow.add(mt);
				}
			}

			sb.append("\n\n<Individual Performances>");
			sb.append("\nID");
			for (MetricType mt : toShow) {
				sb.append(String.format("\t%s", mt));
				if (toShowImprs.contains(mt)) {
					sb.append(String.format("\tImpr"));
				}
			}

			sb.append("\n");

			for (String qid : qids) {
				sb.append(qid);
				for (MetricType mt : toShow) {
					double score = queryScores.getCount(mt, qid);
					double impr = queryImprs.getCount(mt, qid);

					if (toShowImprs.contains(mt)) {
						append(sb, score);
						append(sb, impr);
					} else {
						append(sb, score);
					}
				}
				sb.append("\n");
			}
		}

		return sb.toString().trim();
	}

	private void writeCounter(ObjectOutputStream oos, Counter<MetricType> c) throws Exception {
		oos.writeInt(c.size());
		for (MetricType mt : c.keySet()) {
			oos.writeInt(mt.ordinal());
			oos.writeDouble(c.getCount(mt));
		}
	}

	private void writeCounterMap(ObjectOutputStream oos, CounterMap<MetricType, String> cm) throws Exception {
		oos.writeInt(cm.size());
		for (MetricType mt : cm.keySet()) {
			oos.writeInt(mt.ordinal());
			FileUtils.writeStringCounter(oos, cm.getCounter(mt));
		}
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(top_n);
		oos.writeUTF(name);
		writeCounter(oos, scores);
		writeCounter(oos, imprs);
		writeCounter(oos, ris);
		writeCounterMap(oos, queryScores);
		writeCounterMap(oos, queryImprs);
	}

	public void writeObject(String fileName) throws Exception {
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}
}
