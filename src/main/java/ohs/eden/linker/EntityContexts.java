package ohs.eden.linker;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Map;
import java.util.Map.Entry;

import ohs.io.FileUtils;
import ohs.math.VectorMath;
import ohs.matrix.SparseVector;
import ohs.types.generic.Indexer;
import ohs.utils.Generics;
import ohs.utils.Timer;

public class EntityContexts {
	private Map<Integer, SparseVector> contVecs;

	private Indexer<String> wordIndexer;

	public EntityContexts() {
		wordIndexer = new Indexer<String>();
		contVecs = Generics.newHashMap();
	}

	public EntityContexts(Indexer<String> wordIndexer, Map<Integer, SparseVector> contVecs) {
		this.wordIndexer = wordIndexer;
		this.contVecs = contVecs;
	}

	public Map<Integer, SparseVector> getContextVectors() {
		return contVecs;
	}

	public Indexer<String> getWordIndexer() {
		return wordIndexer;
	}

	public void read(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		Timer timer = Timer.newTimer();
		timer.start();

		wordIndexer = FileUtils.readStringIndexer(ois);
		int size = ois.readInt();
		for (int i = 0; i < size; i++) {
			int id = ois.readInt();
			SparseVector sv = new SparseVector();
			sv.readObject(ois);
			contVecs.put(id, sv);

			VectorMath.unitVector(sv);
		}
		System.out.printf("read [%s] - [%s]\n", this.getClass().getName(), timer.stop());
	}

	public void setContextVectors(Map<Integer, SparseVector> conVecs) {
		this.contVecs = conVecs;
	}

	public void setWordIndexer(Indexer<String> wordIndexer) {
		this.wordIndexer = wordIndexer;
	}

	public void write(String fileName) throws Exception {
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		Timer timer = Timer.newTimer();
		timer.start();

		FileUtils.writeStringIndexer(oos, wordIndexer);
		oos.writeInt(contVecs.size());
		for (Entry<Integer, SparseVector> e : contVecs.entrySet()) {
			oos.writeInt(e.getKey());
			e.getValue().writeObject(oos);
		}

		System.out.printf("write [%s] - [%s]\n", this.getClass().getName(), timer.stop());
	}
}
