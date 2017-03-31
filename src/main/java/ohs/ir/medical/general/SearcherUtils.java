package ohs.ir.medical.general;

import java.nio.file.Paths;
import java.util.Map;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.store.FSDirectory;

import ohs.io.TextFileWriter;
import ohs.ir.lucene.common.AnalyzerUtils;
import ohs.ir.lucene.common.CommonFieldNames;
import ohs.ir.lucene.common.MedicalEnglishAnalyzer;
import ohs.math.VectorUtils;
import ohs.matrix.SparseVector;
import ohs.types.generic.Indexer;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class SearcherUtils {

	public static IndexSearcher getIndexSearcher(String indexDirName) throws Exception {
		System.out.printf("open an index at [%s]\n", indexDirName);
		IndexSearcher ret = new IndexSearcher(DirectoryReader.open(FSDirectory.open(Paths.get(indexDirName))));
		ret.setSimilarity(new LMDirichletSimilarity());
		// indexSearcher.setSimilarity(new BM25Similarity());
		// indexSearcher.setSimilarity(new DFRSimilarity(new BasicModelBE(), new
		// AfterEffectB(), new NormalizationH1()));
		return ret;
	}

	public static IndexSearcher[] getIndexSearchers(String[] indexDirNames) throws Exception {
		IndexSearcher[] ret = new IndexSearcher[indexDirNames.length];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = getIndexSearcher(indexDirNames[i]);
		}
		return ret;
	}

	public static QueryParser getQueryParser() throws Exception {
		QueryParser ret = new QueryParser(CommonFieldNames.CONTENT, MedicalEnglishAnalyzer.newAnalyzer());
		return ret;
	}

	public static SparseVector search(Query q, IndexSearcher is, int top_k) throws Exception {
		TopDocs topDocs = is.search(q, top_k);
		int num_docs = topDocs.scoreDocs.length;

		SparseVector ret = new SparseVector(num_docs);
		for (int i = 0; i < topDocs.scoreDocs.length; i++) {
			ScoreDoc scoreDoc = topDocs.scoreDocs[i];
			ret.addAt(i, scoreDoc.doc, scoreDoc.score);
		}
		ret.sortIndexes();
		return ret;
	}

	public static SparseVector search(Query q, IndexSearcher is, int top_k, Map<Integer, String> docIdMap) throws Exception {
		TopDocs topDocs = is.search(q, top_k);
		int doc_size = topDocs.scoreDocs.length;

		SparseVector ret = new SparseVector(doc_size);
		for (int i = 0; i < topDocs.scoreDocs.length; i++) {
			int docid = topDocs.scoreDocs[i].doc;
			ScoreDoc scoreDoc = topDocs.scoreDocs[i];
			Document doc = is.doc(docid);
			docIdMap.put(docid, doc.get(CommonFieldNames.DOCUMENT_ID));
			ret.addAt(i, scoreDoc.doc, scoreDoc.score);
		}
		ret.sortIndexes();
		return ret;
	}

	public static SparseVector search(SparseVector qm, Indexer<String> wordIndexer, IndexSearcher is, int top_k) throws Exception {
		Query q = AnalyzerUtils.getQuery(VectorUtils.toCounter(qm, wordIndexer));
		return search(q, is, top_k);
	}

	public static void write(TextFileWriter writer, String qid, SparseVector docScores) {
		docScores.sortValues();
		for (int i = 0; i < docScores.size(); i++) {
			int docId = docScores.indexAt(i);
			double score = docScores.valueAt(i);
			writer.write(qid + "\t" + docId + "\t" + score + "\n");
		}
		docScores.sortIndexes();
	}

	public static void write(TextFileWriter writer, String qid, SparseVector docScores, Map<Integer, String> docIdMap) {
		docScores.sortValues();
		for (int i = 0; i < docScores.size(); i++) {
			int indexid = docScores.indexAt(i);
			String docid = docIdMap.get(indexid);
			double score = docScores.valueAt(i);
			writer.write(String.format("%s\t%d\t%s\t%f\n", qid, indexid, docid, score));
		}
		docScores.sortIndexes();
	}

}
