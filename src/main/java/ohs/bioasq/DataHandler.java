package ohs.bioasq;

import java.io.File;
import java.io.ObjectOutputStream;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.corpus.search.DocumentStore;
import ohs.corpus.type.DocumentCollection;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.search.app.DocumentSearcher;
import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListMap;
import ohs.types.generic.Pair;
import ohs.types.generic.SetMap;
import ohs.types.generic.Triple;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class DataHandler {
	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		DataHandler dh = new DataHandler();
		// dh.classifyYearsAndJournals();
		// dh.classifyMeshes();
		// dh.classifyTopNodes();
		// dh.buildTopMeshClassifierData();
		// dh.buildDocumentEmbeddings();
//		dh.buildJournalEmbeddings();
//		dh.test();
		
		System.out.println(Byte.SIZE);
		System.out.println(Short.SIZE);

		System.out.println("process ends.");
	}

	public void test() throws Exception {
		Indexer<String> journalIdxer = FileUtils.readStringIndexerFromText(MIRPath.BIOASQ_DIR + "doc-class/journal/journal_indexer.txt.gz");
		RandomAccessDenseMatrix ED = new RandomAccessDenseMatrix(MIRPath.BIOASQ_DIR + "emb-journal.ser.gz", false);

		// for (String jnl : journalIdxer.getObjects()) {
		// System.out.println(jnl);
		// }

		for (int i = 0; i < 10; i++) {

			Counter<Integer> c = Generics.newCounter();
			for (int j = 0; j < ED.rowSize(); j++) {
				double cosine = VectorMath.cosine(ED.row(i), ED.row(j));
				c.setCount(j, cosine);
			}

			String jnl = journalIdxer.getObject(i);

			System.out.printf("%s, %s\n", jnl, VectorUtils.toCounter(c, journalIdxer));
		}

	}

	public void buildTopMeshClassifierData() throws Exception {
		MeshTree mt = new MeshTree(MIRPath.BIOASQ_MESH_TREE_SER_FILE);

		String dir = MIRPath.BIOASQ_DIR + "doc-class/top-mesh/";

		Map<Integer, IntegerArray> topToDoc = Generics.newHashMap();

		for (File file : FileUtils.getFilesUnder(dir)) {
			String fileName = file.getName();
			int idx = fileName.indexOf(".");
			fileName = fileName.substring(0, idx);
			int top = Integer.parseInt(fileName);
			topToDoc.put(top, new IntegerArray(file.getPath()));
		}

		Set<Integer> set = Generics.newHashSet();

		for (IntegerArray seqs : topToDoc.values()) {
			for (int docseq : seqs) {
				set.add(docseq);
			}
		}

		IntegerArray docseqs = new IntegerArray(set);
		docseqs.sort(false);

		IntegerArray meshNums = new IntegerArray(docseqs.size());
		IntegerArray journals = new IntegerArray(docseqs.size());

		DocumentSearcher ds = new DocumentSearcher(MIRPath.BIOASQ_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);
		Vocab vocab = ds.getVocab();

		DenseMatrix EW = new RandomAccessDenseMatrix(MIRPath.BIOASQ_DIR + "glove_model_raf.ser").rowsAsMatrix();

		// DenseVector idfs = new DenseVector(E.rowSize());

		// for (int i = 0; i < ds.getVocab().size(); i++) {
		// double idf = TermWeighting.idf(vocab.getDocCnt(), vocab.getDocFreq(i));
		// idfs.add(i, idf);
		// }
		//
		// for (int i = 0; i < E.rowSize(); i++) {
		// DenseVector e = E.row(i);
		// e.multiply(idfs.prob(i));
		// }

		DenseMatrix ED = new DenseMatrix(docseqs.size(), EW.colSize());

		for (int i = 0; i < docseqs.size(); i++) {
			int docseq = docseqs.get(i);
			Pair<String, IntegerArrayMatrix> t = ds.getDocumentCollection().getSents(docseq);
			String s = t.getThird();

			String[] items = StrUtils.unwrap(s.split("\t"));

			String journal = items[0];
			String year = items[1];
			String[] meshes = items[2].split("\\|");

			SparseVector dv = ds.getDocumentCollection().getVector(docseq);

			DenseVector ed = ED.row(i);

			for (int j = 0; j < dv.size(); j++) {
				int w = dv.indexAt(j);
				double cnt = dv.valueAt(j);
				DenseVector ew = EW.row(w);
				ArrayMath.addAfterMultiply(ew.values(), cnt, ed.values());
			}

			ed.multiply(1f / dv.sum());

			meshNums.add(meshes.length);
		}

		ED.writeObject(MIRPath.BIOASQ_DIR + "top-doc-embs.ser.gz");
		docseqs.writeObject(MIRPath.BIOASQ_DIR + "top-docseqs.ser.gz");
		meshNums.writeObject(MIRPath.BIOASQ_DIR + "top-mesh-nums.ser.gz");
	}

	public void buildDocumentEmbeddings() throws Exception {

		String tmpFileName = MIRPath.BIOASQ_DIR + "emb-doc-tmp.ser.gz";

		FileUtils.delete(tmpFileName);

		if (!FileUtils.exists(tmpFileName)) {
			DocumentSearcher ds = new DocumentSearcher(MIRPath.BIOASQ_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);
			DocumentCollection store = ds.getDocumentCollection();

			DenseMatrix EW = new RandomAccessDenseMatrix(MIRPath.BIOASQ_DIR + "glove_model_raf.ser").rowsAsMatrix();
			ObjectOutputStream oos = FileUtils.openObjectOutputStream(MIRPath.BIOASQ_DIR + "emb-doc-tmp.ser.gz");
			oos.writeInt(store.getBytes());
			oos.writeInt(EW.colSize());

			Timer timer = Timer.newTimer();

			for (int i = 0; i < store.getBytes(); i++) {
				if ((i + 1) % 100000 == 0) {
					System.out.printf("[%d/%d, %s]\n", i + 1, store.getBytes(), timer.stop());
				}

				Triple<String, IntegerArrayMatrix, String> t = store.getSents(i);
				String s = t.getThird();

				String[] items = StrUtils.unwrap(s.split("\t"));

				String journal = items[0];
				String year = items[1];
				String[] meshes = items[2].split("\\|");

				SparseVector dv = ds.getDocumentCollection().getVector(i);

				DenseVector ed = new DenseVector(EW.colSize());

				for (int j = 0; j < dv.size(); j++) {
					int w = dv.indexAt(j);
					double cnt = dv.valueAt(j);
					DenseVector ew = EW.row(w);
					ArrayMath.addAfterMultiply(ew.values(), cnt, ed.values());
				}

				ed.multiply(1f / dv.sum());
				ed.writeObject(oos);
			}
			System.out.printf("[%d/%d, %s]\n", store.getBytes(), store.getBytes(), timer.stop());
			oos.close();
		}

		RandomAccessDenseMatrix.buildFromTempFile(tmpFileName, tmpFileName.replace("-tmp", ""));
	}

	public void buildJournalEmbeddings() throws Exception {
		Indexer<String> journalIdxer = FileUtils.readStringIndexerFromText(MIRPath.BIOASQ_DIR + "doc-class/journal/journal_indexer.txt.gz");

		RandomAccessDenseMatrix ED = new RandomAccessDenseMatrix(MIRPath.BIOASQ_DIR + "emb-doc.ser.gz", false);

		List<File> files = FileUtils.getFilesUnder(MIRPath.BIOASQ_DIR + "doc-class/journal");

		Map<Integer, File> m = Generics.newHashMap();

		for (File file : files) {
			if (file.getPath().contains("txt.gz")) {
				continue;
			}

			String fileName = file.getName();
			int idx = fileName.indexOf(".");
			fileName = fileName.substring(0, idx);
			int jnl = Integer.parseInt(fileName);
			m.put(jnl, file);
		}

		List<Integer> jnls = Generics.newArrayList(m.keySet());
		Collections.sort(jnls);

		String tmpFileName = MIRPath.BIOASQ_DIR + "emb-journal-tmp.ser.gz";

		// if (!FileUtils.exists(tmpFileName)) {
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(tmpFileName);
		oos.writeInt(jnls.size());
		oos.writeInt(ED.colSize());

		for (int i = 0; i < journalIdxer.size(); i++) {
			File file = m.get(i);
			DenseVector ej = new DenseVector(ED.colSize());

			if (file.exists()) {
				IntegerArray docseqs = new IntegerArray(file.getPath());
				for (int docseq : docseqs) {
					VectorMath.add(ED.row(docseq), ej);
				}
				ej.multiply(1f / docseqs.size());
			} else {
				System.out.println(file);
			}
			ej.writeObject(oos);
		}
		oos.close();
		// }

		RandomAccessDenseMatrix.buildFromTempFile(tmpFileName, tmpFileName.replace("-tmp", ""));

	}

	public void classifyMeshes() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(MIRPath.BIOASQ_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);

		MeshTree mt = new MeshTree(MIRPath.BIOASQ_MESH_TREE_SER_FILE);

		SetMap<Integer, Integer> topNodeMap = Generics.newSetMap();

		for (int i = 0; i < mt.size(); i++) {
			for (IntegerArray path : mt.getPaths(i)) {
				int top_node = path.get(1);
				topNodeMap.put(i, top_node);
			}
		}

		DocumentCollection store = ds.getDocumentCollection();

		ListMap<Integer, Integer> meshToDoc = Generics.newListMap();

		for (int i = 0; i < store.getBytes(); i++) {
			if ((i + 1) % 100000 == 0) {
				System.out.printf("[%d/%d]\n", i + 1, store.getBytes());
			}

			Triple<String, IntegerArrayMatrix, String> t = store.getSents(i);
			String s = t.getThird();

			String[] items = s.split("\t");

			items = StrUtils.unwrap(items);

			String journal = items[0];
			String year = items[1];
			String[] meshes = items[2].split("\\|");

			for (String mesh : meshes) {
				int mesh_idx = mt.indexOfName(mesh);

				if (mesh_idx < 0) {
					System.out.printf("not found [%s]\n", meshes);
				}
				meshToDoc.put(mesh_idx, i);
			}
		}
		System.out.printf("[%d/%d]\n", store.getBytes(), store.getBytes());

		{
			List<Integer> keys = Generics.newArrayList(meshToDoc.keySet());
			Collections.sort(keys);
			Collections.reverse(keys);

			String outDir = MIRPath.BIOASQ_DIR + "doc-class/mesh/";

			for (int i = 0; i < keys.size(); i++) {
				int year = keys.get(i);
				IntegerArray docseqs = new IntegerArray(meshToDoc.get(year));
				docseqs.sort(false);

				docseqs.writeObject(String.format("%s%d.ser.gz", outDir, year));
			}
		}
	}

	public void classifyTopNodes() throws Exception {
		MeshTree mt = new MeshTree(MIRPath.BIOASQ_MESH_TREE_SER_FILE);
		SetMap<Integer, Integer> sm = Generics.newSetMap();

		for (int i = 0; i < mt.size(); i++) {
			for (IntegerArray path : mt.getPaths(i)) {
				sm.put(i, path.get(1));
			}
		}

		String dir = MIRPath.BIOASQ_DIR + "doc-class/year/";

		IntegerArrayMatrix m = new IntegerArrayMatrix();
		m.add(new IntegerArray(dir + "2014.ser.gz"));
		m.add(new IntegerArray(dir + "2015.ser.gz"));
		m.add(new IntegerArray(dir + "2016.ser.gz"));

		DocumentSearcher ds = new DocumentSearcher(MIRPath.BIOASQ_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);

		ListMap<Integer, Integer> lm = Generics.newListMap();

		for (IntegerArray docseqs : m) {
			for (int docseq : docseqs) {

				Triple<String, IntegerArrayMatrix, String> t = ds.getDocumentCollection().getSents(docseq);
				String s = t.getThird();

				String[] items = s.split("\t");

				items = StrUtils.unwrap(items);

				String journal = items[0];
				String year = items[1];
				String[] meshs = items[2].split("\\|");

				for (String mesh : meshs) {
					int mesh_idx = mt.indexOfName(mesh);
					Set<Integer> topNodes = sm.get(mesh_idx);
					for (int top_node : topNodes) {
						lm.put(top_node, docseq);
					}
				}
			}
		}

		for (int top_node : lm.keySet()) {
			IntegerArray docseqs = new IntegerArray(lm.get(top_node));
			docseqs.writeObject(String.format("%s%d.ser.gz", MIRPath.BIOASQ_DIR + "doc-class/top-mesh/", top_node));
		}

	}

	public void classifyYearsAndJournals() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(MIRPath.BIOASQ_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);

		MeshTree mt = new MeshTree(MIRPath.BIOASQ_MESH_TREE_SER_FILE);

		Indexer<String> journalIndexer = Generics.newIndexer();

		SetMap<Integer, Integer> topNodeMap = Generics.newSetMap();

		for (int i = 0; i < mt.size(); i++) {
			for (IntegerArray path : mt.getPaths(i)) {
				int top_node = path.get(1);
				topNodeMap.put(i, top_node);
			}
		}

		DocumentStore store = ds.getDocumentCollection();

		ListMap<Integer, Integer> yearToDoc = Generics.newListMap();
		ListMap<Integer, Integer> journalToDoc = Generics.newListMap();

		for (int i = 0; i < store.getBytes(); i++) {
			if ((i + 1) % 100000 == 0) {
				System.out.printf("[%d/%d]\n", i + 1, store.getBytes());
			}

			Triple<String, IntegerArrayMatrix, String> t = store.getSents(i);
			String s = t.getThird();

			String[] items = s.split("\t");

			items = StrUtils.unwrap(items);

			String journal = items[0];
			String year = items[1];
			String[] meshs = items[2].split("\\|");

			if (year.length() == 0) {
				year = "1900";
			}

			int jid = journalIndexer.getIndex(journal);

			yearToDoc.put(Integer.parseInt(year), i);
			journalToDoc.put(jid, i);
		}
		System.out.printf("[%d/%d]\n", store.getBytes(), store.getBytes());

		{
			List<Integer> keys = Generics.newArrayList(yearToDoc.keySet());
			Collections.sort(keys);

			String outDir = MIRPath.BIOASQ_DIR + "doc-class/year/";

			FileUtils.deleteFilesUnder(outDir);

			for (int i = 0; i < keys.size(); i++) {
				int year = keys.get(i);
				IntegerArray docseqs = new IntegerArray(yearToDoc.get(year));
				docseqs.sort(false);

				docseqs.writeObject(String.format("%s%d.ser.gz", outDir, year));
			}
		}

		{
			List<Integer> keys = Generics.newArrayList(journalToDoc.keySet());
			Collections.sort(keys);

			String outDir = MIRPath.BIOASQ_DIR + "doc-class/journal/";

			FileUtils.deleteFilesUnder(outDir);

			for (int i = 0; i < keys.size(); i++) {
				int jid = keys.get(i);
				IntegerArray docseqs = new IntegerArray(journalToDoc.get(jid));
				docseqs.sort(false);

				docseqs.writeObject(String.format("%s%s.ser.gz", outDir, jid));
			}

			FileUtils.writeStringIndexerAsText(outDir + "journal_indexer.txt.gz", journalIndexer);
		}
	}

}
