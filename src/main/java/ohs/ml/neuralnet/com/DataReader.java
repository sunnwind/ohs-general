package ohs.ml.neuralnet.com;

import java.util.List;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.Indexer;
import ohs.types.generic.Pair;
import ohs.types.generic.Triple;
import ohs.types.generic.Vocab;
import ohs.types.number.DoubleArray;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class DataReader {

	public static Triple<IntegerMatrix, IntegerMatrix, Vocab> readCapitalData(int size) throws Exception {
		List<String> lines = Generics.newLinkedList();
		for (String line : FileUtils.readLinesFromText("capitals.txt")) {
			String[] parts = line.split("\t");
			if (parts.length != 2) {
				continue;
			}

			for (String s : parts) {
				lines.add(s);
			}
		}

		Vocab vocab = new Vocab();

		List<IntegerArray> X = Generics.newLinkedList();
		List<IntegerArray> Y = Generics.newLinkedList();

		for (int i = 0; i < lines.size() && i < size; i++) {
			String s = lines.get(i).toLowerCase();
			s = "<" + s + ">";
			int[] t = new int[s.length()];

			for (int j = 0; j < t.length; j++) {
				t[j] = vocab.getIndex(s.charAt(j) + "");
			}

			int[] x = new int[t.length - 1];
			int[] y = new int[t.length - 1];

			for (int j = 0; j < x.length; j++) {
				x[j] = t[j];
				y[j] = t[j + 1];
			}

			X.add(new IntegerArray(x));
			Y.add(new IntegerArray(y));
		}

		return Generics.newTriple(new IntegerMatrix(X), new IntegerMatrix(Y), vocab);
	}

	public static Pair<SparseMatrix, IntegerArray> readFromSvmFormat(String fileName) throws Exception {
		List<Integer> rowIdxs = Generics.newLinkedList();
		List<SparseVector> rows = Generics.newLinkedList();
		IntegerArray labels = new IntegerArray();

		for (String line : FileUtils.readLinesFromText(fileName)) {
			String[] parts = line.split(" ");

			IntegerArray idxs = new IntegerArray();
			DoubleArray vals = new DoubleArray();

			for (int i = 1; i < parts.length; i++) {
				String[] toks = parts[i].split(":");
				idxs.add(Integer.parseInt(toks[0]) - 1);
				vals.add(Double.parseDouble(toks[1]));
			}

			idxs.trimToSize();
			vals.trimToSize();

			int y = Integer.parseInt(parts[0]);
			labels.add(y);

			rowIdxs.add(rowIdxs.size());
			rows.add(new SparseVector(idxs.values(), vals.values()));
		}

		labels.trimToSize();

		return Generics.newPair(new SparseMatrix(rowIdxs, rows), labels);
	}

	public static Triple<IntegerMatrix, IntegerMatrix, Vocab> readLines(String fileName, int size) throws Exception {
		List<String> lines = Generics.newLinkedList();
		for (String line : FileUtils.readLinesFromText(fileName)) {
			if (lines.size() == size) {
				break;
			}
			line = StrUtils.normalizeSpaces(line);
			lines.add(line);
		}

		Vocab vocab = new Vocab();

		List<IntegerArray> X = Generics.newLinkedList();
		List<IntegerArray> Y = Generics.newLinkedList();

		for (int i = 0; i < lines.size() && i < size; i++) {
			String str = lines.get(i).toLowerCase();
			str = "<" + str + ">";
			IntegerArray s = new IntegerArray(str.length() + 2);

			for (int j = 0; j < str.length(); j++) {
				s.add(vocab.getIndex(str.charAt(j) + ""));
			}

			IntegerArray x = new IntegerArray();
			IntegerArray y = new IntegerArray();

			for (int j = 0; j < s.size() - 1; j++) {
				x.add(s.get(j));
				y.add(s.get(j + 1));
			}

			x.trimToSize();
			y.trimToSize();

			X.add(x);
			Y.add(y);
		}

		return Generics.newTriple(new IntegerMatrix(X), new IntegerMatrix(Y), vocab);
	}

	public static Object[] readNerTestData(String fileName, Vocab vocab, Indexer<String> labelIndexer) throws Exception {
		List<IntegerArray> X = Generics.newLinkedList();
		List<IntegerArray> Y = Generics.newLinkedList();

		for (String p : FileUtils.readFromText(fileName).split("\n\n")) {
			String[] lines = p.split("\n");

			IntegerArray x = new IntegerArray();
			IntegerArray y = new IntegerArray();

			for (int j = 0; j < lines.length; j++) {
				String line = lines[j];
				String[] parts = line.split(" ");
				String word = parts[0];
				String pos = parts[1];
				String phrase = parts[2];
				String ner = parts[3];
				x.add(vocab.indexOf(word, 0));
				y.add(labelIndexer.indexOf(ner));
			}
			X.add(x);
			Y.add(y);
		}
		return new Object[] { new IntegerMatrix(X), new IntegerMatrix(Y) };
	}

	public static Object[] readNerTrainData(String fileName) throws Exception {
		List<IntegerArray> X = Generics.newLinkedList();
		List<IntegerArray> Y = Generics.newLinkedList();

		Vocab vocab = new Vocab();
		vocab.add(Vocab.SYM.UNK.getText());

		Indexer<String> labelIndexer = Generics.newIndexer();
		Set<String> labels = Generics.newTreeSet();

		String[] ps = FileUtils.readFromText(fileName).split("\n\n");

		for (String p : ps) {
			String[] lines = p.split("\n");
			for (int j = 0; j < lines.length; j++) {
				String line = lines[j];
				String[] parts = line.split(" ");
				String word = parts[0];
				String pos = parts[1];
				String phrase = parts[2];
				String ner = parts[3];
				labels.add(ner);
			}
		}

		labelIndexer.addAll(labels);

		for (String p : ps) {
			String[] lines = p.split("\n");

			IntegerArray x = new IntegerArray();
			IntegerArray y = new IntegerArray();

			for (int j = 0; j < lines.length; j++) {
				String line = lines[j];
				String[] parts = line.split(" ");
				String word = parts[0];
				String pos = parts[1];
				String phrase = parts[2];
				String ner = parts[3];
				x.add(vocab.getIndex(word.toLowerCase()));
				y.add(labelIndexer.indexOf(ner));
			}
			X.add(x);
			Y.add(y);
		}

		return new Object[] { new IntegerMatrix(X), new IntegerMatrix(Y), vocab, labelIndexer };
	}

}
