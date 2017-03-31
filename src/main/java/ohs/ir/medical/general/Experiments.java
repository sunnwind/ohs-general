package ohs.ir.medical.general;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;

import ohs.eden.linker.Entity;
import ohs.eden.linker.EntityLinker;
import ohs.io.FileUtils;
import ohs.io.TextFileWriter;
import ohs.ir.lucene.common.AnalyzerUtils;
import ohs.ir.lucene.common.CommonFieldNames;
import ohs.ir.lucene.common.MedicalEnglishAnalyzer;
import ohs.ir.medical.query.BaseQuery;
import ohs.ir.medical.query.QueryReader;
import ohs.ir.wiki.WikiXmlDataHandler;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.math.LA;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.ml.word2vec.Word2VecModel;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class Experiments {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		Experiments e = new Experiments();
		e.searchByQLD();
		// e.searchByKLD();
		// e.searchByKLDFB();
		// e.searchByKldFbPriors();
		// e.searchByCBEEM();
		// e.searchSentsByQLD();
		// e.searchSentsByKLDFB();

		// e.searchByKldFbWordVectors();
		// e.searchByKldFbWordVectorExp2();
		// e.searchByKldFbWordVectorExp();
		// e.searchByKldFbWordVectorPrior();
		// e.searchByEntityLinking();
		// e.testKLDFBWiki();
		// e.searchByKLDFBWiki();
		 SearchResultEvaluator.main(args);

		System.out.println("process ends.");
	}

	private MedicalEnglishAnalyzer analyzer = MedicalEnglishAnalyzer.newAnalyzer();

	private String[] queryFileNames = MIRPath.QueryFileNames;

	private String[] indexDirNames = MIRPath.IndexDirNames;

	private String[] resDirNames = MIRPath.ResultDirNames;

	private String[] docIdMapFileNames = MIRPath.DocIdMapFileNames;

	private String[] relFileNames = MIRPath.RelevanceFileNames;

	private IndexSearcher[] iss;

	public Experiments() throws Exception {
		iss = new IndexSearcher[indexDirNames.length];

		Map<String, Integer> map = Generics.newHashMap();

		for (int i = 0; i < indexDirNames.length; i++) {
			Integer idx = map.get(indexDirNames[i]);
			if (idx == null) {
				map.put(indexDirNames[i], i);
				iss[i] = SearcherUtils.getIndexSearcher(indexDirNames[i]);
			} else {
				iss[i] = iss[idx.intValue()];
			}
		}
	}

	public Counter<String> expand(IndexReader ir, Word2VecSearcher searcher, Counter<String> wcs) throws IOException {

		Counter<String> wwc = new Counter<String>();
		double norm = 0;

		for (String word : wcs.keySet()) {
			double cnt = wcs.getCount(word);
			double tf = Math.log(cnt) + 1;
			double doc_freq = ir.docFreq(new Term(CommonFieldNames.CONTENT, word));
			double num_docs = ir.maxDoc() + 1;
			double idf = Math.log((num_docs + 1) / doc_freq);
			double tfidf = tf * idf;
			wwc.setCount(word, tfidf);
			norm += (tfidf * tfidf);
		}

		norm = Math.sqrt(norm);
		wwc.scale(1f / norm);

		List<String> words = wwc.getSortedKeys();
		//
		// Counter<String>Map cm = new Counter<String>Map();
		// for (int i = 0; i < words.size() && i < 5; i++) {
		// Counter<String> c = searcher.search(words.get(i));
		// cm.setCounter(words.get(i), c);
		// }

		double[] qwv = new double[searcher.getLayerSize()];
		double[][] vectors = searcher.getVectors();

		for (String word : wwc.getSortedKeys()) {
			double[] v = searcher.getVector(word);
			if (v != null) {
				double tfidf = wwc.getCount(word);
				ArrayMath.addAfterMultiply(qwv, 1, v, tfidf, qwv);
			}

			Counter<String> c = new Counter<>();

			for (int i = 0; i < searcher.getWordIndexer().size(); i++) {
				c.incrementCount(searcher.getWordIndexer().getObject(i), ArrayMath.dotProduct(v, vectors[i]));
			}

			System.out.printf("%s -> %s\n", word, c.toString());
		}

		ArrayMath.unitVector(qwv, qwv);

		double[][] wvs = searcher.getVectors();
		Counter<String> c = new Counter<String>();

		int dim = wvs.length;

		for (int i = 0; i < wvs.length; i++) {
			String word = searcher.getWordIndexer().getObject(i);
			if (word.contains("<N")) {
				continue;
			}
			double sim = ArrayMath.dotProduct(qwv, wvs[i]);
			c.incrementCount(word, sim);
		}

		Counter<String> ret = new Counter<String>(wcs);

		words = c.getSortedKeys();
		for (int i = 0, j = 0; i < words.size(); i++) {
			String word = words.get(i);
			double sim = c.getCount(word);
			if (sim < 0.5) {
				break;
			}

			if (ret.containsKey(word)) {
				continue;
			}

			if (j++ == 5) {
				break;
			}

			ret.incrementCount(word, 1);
		}

		System.out.println(wcs.toString());
		System.out.println(c.toString());
		System.out.println(ret.toString());
		System.out.println();

		return ret;
	}

	public double[] getVectorSum(IndexReader ir, Word2VecSearcher searcher, Counter<String> wcs) throws Exception {
		double[] ret = new double[searcher.getLayerSize()];

		double norm = 0;

		for (String word : wcs.keySet()) {
			double cnt = wcs.getCount(word);
			double tf = Math.log(cnt) + 1;
			double doc_freq = ir.docFreq(new Term(CommonFieldNames.CONTENT, word));
			double num_docs = ir.maxDoc() + 1;
			double idf = Math.log((num_docs + 1) / doc_freq);
			double tfidf = tf * idf;
			wcs.setCount(word, tfidf);

			norm += (tfidf * tfidf);
		}

		norm = Math.sqrt(norm);
		wcs.scale(1f / norm);

		int num_words = 0;

		List<String> words = wcs.getSortedKeys();

		for (int i = 0; i < words.size(); i++) {
			String word = words.get(i);
			double tfidf = wcs.getCount(word);
			double[] v = searcher.getVector(word);
			if (v != null) {
				double sum = ArrayMath.addAfterMultiply(ret, 1, v, tfidf, ret);
				num_words++;
			}
		}

		ArrayMath.multiply(ret, 1f / num_words, ret);

		return ret;
	}

	public void searchByCBEEM() throws Exception {
		System.out.println("search by CBEEM.");

		DenseVector[] docPriorData = new DenseVector[iss.length];

		for (int i = 0; i < indexDirNames.length; i++) {
			File inputFile = null;
			DenseVector docPriors = null;
			if (inputFile != null && inputFile.exists()) {
				docPriors = new DenseVector();
				docPriors.readObject(inputFile.getPath());

				double uniform_prior = 1f / docPriors.size();
				for (int j = 0; j < docPriors.size(); j++) {
					if (docPriors.value(j) == 0) {
						docPriors.set(j, uniform_prior);
					}
				}
			} else {
				docPriors = new DenseVector(iss[i].getIndexReader().maxDoc());
				double uniform_prior = 1f / docPriors.size();
				docPriors.setAll(uniform_prior);
			}
			docPriorData[i] = docPriors;
		}

		HyperParameter hp = new HyperParameter();

		for (int i = 0; i < queryFileNames.length; i++) {
			List<BaseQuery> bqs = QueryReader.readQueries(queryFileNames[i]);
			String outputFileName = resDirNames[i] + "cbeem.txt";
			CbeemDocumentSearcher cbeemSearcher = new CbeemDocumentSearcher(iss, docPriorData, hp, analyzer, false);
			cbeemSearcher.search(i, bqs, null, outputFileName, null);
		}

	}

	public void searchByEntityLinking() throws Exception {
		String dirPath = "../../data/entity_iden/wiki/el/";

		EntityLinker el = new EntityLinker();
		el.read("../../data/entity_iden/wiki/entity-linker_all.ser.gz");
		el.setTopK(10);

		for (int i = 0; i < queryFileNames.length; i++) {
			List<BaseQuery> bqs = QueryReader.readQueries(queryFileNames[i]);

			String outputFileName = String.format("%s/%d.txt", dirPath, i);

			TextFileWriter writer = new TextFileWriter(outputFileName);

			for (int j = 0; j < bqs.size(); j++) {
				BaseQuery bq = bqs.get(j);
				List<String> words = StrUtils.split(bq.getSearchText());

				Counter<Entity> candidates = new Counter<Entity>();

				for (String ngram : StrUtils.ngrams(2, words).keySet()) {
					candidates.incrementAll(el.link(ngram.replace("_", " ")));
				}

				writer.write(bq.toString() + "\n");
				writer.write(candidates.toStringSortedByValues(false, true, 20, " "));
				writer.write("\n\n");

			}
			writer.close();
		}
	}

	public void searchByKLD() throws Exception {
		System.out.println("search by KLD.");

		for (int i = 0; i < queryFileNames.length; i++) {
			List<BaseQuery> bqs = QueryReader.readQueries(queryFileNames[i]);
			IndexSearcher is = iss[i];

			String outputFileName = resDirNames[i] + "kld.txt";

			TextFileWriter writer = new TextFileWriter(outputFileName);

			for (int j = 0; j < bqs.size(); j++) {
				BaseQuery bq = bqs.get(j);
				System.out.println(bq);

				BooleanQuery lbq = AnalyzerUtils.getQuery(bq.getSearchText(), analyzer);
				SparseVector docScores = SearcherUtils.search(lbq, is, 1000);
				docScores.normalizeAfterSummation();

				Indexer<String> wordIndexer = new Indexer<String>();
				Counter qwcs = AnalyzerUtils.getWordCounts(bq.getSearchText(), analyzer);

				SparseVector qlm = VectorUtils.toSparseVector(qwcs, wordIndexer, true);
				qlm.normalize();

				WordCountBox wcb = WordCountBox.getWordCountBox(is, docScores, wordIndexer);

				KLDivergenceScorer scorer = new KLDivergenceScorer();
				docScores = scorer.score(wcb, qlm);

				SearcherUtils.write(writer, bq.getId(), docScores);
			}
			writer.close();
		}

	}

	public void searchByKLDFB() throws Exception {
		System.out.println("search by KLD FB.");

		for (int i = 0; i < queryFileNames.length; i++) {
			List<BaseQuery> bqs = QueryReader.readQueries(queryFileNames[i]);
			IndexSearcher is = iss[i];
			IndexReader ir = is.getIndexReader();

			String outFileName = resDirNames[i] + "kld-fb.txt";

			TextFileWriter writer = new TextFileWriter(outFileName);

			for (int j = 0; j < bqs.size(); j++) {
				BaseQuery bq = bqs.get(j);

				Indexer<String> wordIndexer = new Indexer<String>();
				Counter<String> qwcs = AnalyzerUtils.getWordCounts(bq.getSearchText(), analyzer);

				SparseVector qlm = VectorUtils.toSparseVector(qwcs, wordIndexer, true);
				qlm.normalize();

				SparseVector eqlm = qlm.copy();
				SparseVector docScores = null;

				BooleanQuery lbq = AnalyzerUtils.getQuery(VectorUtils.toCounter(eqlm, wordIndexer));
				docScores = SearcherUtils.search(lbq, is, 1000);

				WordCountBox wcb = WordCountBox.getWordCountBox(is, docScores, wordIndexer, CommonFieldNames.CONTENT);

				RelevanceModelBuilder rmb = new RelevanceModelBuilder();
				SparseVector rm = rmb.getRelevanceModel(wcb, docScores);
				// SparseVector prm = rmb.getPositionalRelevanceModel(qLM, wcb3,
				// docScores);

				double rm_mixture = 0.5;

				eqlm = VectorMath.addAfterMultiply(qlm, 1 - rm_mixture, rm, rm_mixture);

				KLDivergenceScorer scorer = new KLDivergenceScorer();
				docScores = scorer.score(wcb, eqlm);

				System.out.println(bq);
				System.out.printf("QM1:\t%s\n", VectorUtils.toCounter(qlm, wordIndexer));
				System.out.printf("QM2:\t%s\n", VectorUtils.toCounter(eqlm, wordIndexer));

				SearcherUtils.write(writer, bq.getId(), docScores);
			}

			writer.close();
		}
	}

	public void searchByKldFbPriors() throws Exception {
		System.out.println("search by KLD FB Priors.");

		for (int i = 0; i < queryFileNames.length; i++) {
			List<BaseQuery> bqs = QueryReader.readQueries(queryFileNames[i]);
			IndexSearcher is = iss[i];
			IndexReader ir = is.getIndexReader();

			String outputFileName = resDirNames[i] + "kld-fb_prior.txt";

			TextFileWriter writer = new TextFileWriter(outputFileName);

			for (int j = 0; j < bqs.size(); j++) {
				BaseQuery bq = bqs.get(j);

				Indexer<String> wordIndexer = new Indexer<String>();
				Counter<String> qwcs = AnalyzerUtils.getWordCounts(bq.getSearchText(), analyzer);

				SparseVector qlm = VectorUtils.toSparseVector(qwcs, wordIndexer, true);
				qlm.normalize();

				SparseVector eqlm = qlm.copy();
				SparseVector docScores = null;

				BooleanQuery lbq = AnalyzerUtils.getQuery(VectorUtils.toCounter(eqlm, wordIndexer));
				docScores = SearcherUtils.search(lbq, is, 1000);

				WordCountBox wcb = WordCountBox.getWordCountBox(is, docScores, wordIndexer, CommonFieldNames.CONTENT);

				DocumentCentralityEstimator dce = new DocumentCentralityEstimator(wcb);
				SparseVector docPriors = dce.estimate();

				RelevanceModelBuilder rmb = new RelevanceModelBuilder();
				SparseVector rm = rmb.getRelevanceModel(wcb, docScores, docPriors);
				// SparseVector prm = rmb.getPositionalRelevanceModel(qLM, wcb3,
				// docScores);

				double rm_mixture = 0.5;

				eqlm = VectorMath.addAfterMultiply(qlm, 1 - rm_mixture, rm, rm_mixture);

				KLDivergenceScorer scorer = new KLDivergenceScorer();
				docScores = scorer.score(wcb, eqlm);

				System.out.println(bq);
				System.out.printf("QM1:\t%s\n", VectorUtils.toCounter(qlm, wordIndexer));
				System.out.printf("QM2:\t%s\n", VectorUtils.toCounter(eqlm, wordIndexer));

				SearcherUtils.write(writer, bq.getId(), docScores);
			}

			writer.close();
		}
	}

	public void searchByKLDFBWiki() throws Exception {
		System.out.println("search by KLD FB.");
		IndexSearcher wis = SearcherUtils.getIndexSearcher(MIRPath.WIKI_INDEX_DIR);

		Set<String> stopPrefixes = WikiXmlDataHandler.getStopPrefixes();

		for (int i = 0; i < queryFileNames.length; i++) {
			List<BaseQuery> bqs = QueryReader.readQueries(queryFileNames[i]);
			IndexSearcher is = iss[i];
			IndexReader ir = is.getIndexReader();

			String outputFileName = resDirNames[i] + "kld-fb-wiki.txt";

			TextFileWriter writer = new TextFileWriter(outputFileName);

			for (int j = 0; j < bqs.size(); j++) {
				BaseQuery bq = bqs.get(j);

				Indexer<String> wordIndexer = new Indexer<String>();
				Counter<String> qwcs = AnalyzerUtils.getWordCounts(bq.getSearchText(), analyzer);

				SparseVector qlm = VectorUtils.toSparseVector(qwcs, wordIndexer, true);
				qlm.normalize();

				// SparseVector eqlm = qlm.copy();

				BooleanQuery lbq = AnalyzerUtils.getQuery(VectorUtils.toCounter(qlm, wordIndexer));
				SparseVector docScores = SearcherUtils.search(lbq, wis, 50);

				Counter<String> wwcs = Generics.newCounter();

				docScores.sortValues();

				for (int k = 0, l = 0; k < docScores.size() && l < 10; k++) {
					int docid = docScores.indexAt(k);
					double score = docScores.valueAt(k);
					String title = wis.getIndexReader().document(docid).get(CommonFieldNames.TITLE);
					if (WikiXmlDataHandler.accept(stopPrefixes, title)) {
						wwcs.incrementAll(AnalyzerUtils.getWordCounts(title, analyzer));
						l++;
					}
				}

				Counter<String> nqwcs = Generics.newCounter();

				for (String word : qwcs.keySet()) {
					double cnt1 = qwcs.getCount(word);
					double cnt2 = wwcs.getCount(word);
					nqwcs.incrementCount(word, cnt1 + cnt2);
				}

				qlm = VectorUtils.toSparseVector(nqwcs, wordIndexer, true);
				qlm.normalize();

				lbq = AnalyzerUtils.getQuery(VectorUtils.toCounter(qlm, wordIndexer));
				docScores = SearcherUtils.search(lbq, is, 500);

				WordCountBox wcb = WordCountBox.getWordCountBox(is, docScores, wordIndexer, CommonFieldNames.CONTENT);

				RelevanceModelBuilder rmb = new RelevanceModelBuilder();
				SparseVector rm = rmb.getRelevanceModel(wcb, docScores);
				// SparseVector prm = rmb.getPositionalRelevanceModel(qLM, wcb3,
				// docScores);

				double[] mixtures = { 1, 1 };
				ArrayMath.normalize(mixtures);

				SparseVector eqlm = VectorMath.addAfterMultiply(new SparseVector[] { qlm, rm }, mixtures);

				KLDivergenceScorer scorer = new KLDivergenceScorer();
				docScores = scorer.score(wcb, eqlm);

				System.out.println(bq);
				System.out.printf("QM1:\t%s\n", VectorUtils.toCounter(qlm, wordIndexer));
				System.out.printf("QM2:\t%s\n", VectorUtils.toCounter(eqlm, wordIndexer));

				SearcherUtils.write(writer, bq.getId(), docScores);

			}

			writer.close();
		}
	}

	public void searchByKldFbWordVectorExp() throws Exception {
		System.out.println("search by KLD FB Word Vector Exp.");

		Word2VecSearcher vSearcher = new Word2VecSearcher(
				Word2VecModel.fromSerFile("../../data/medical_ir/ohsumed/word2vec_model_stem.ser.gz"));

		for (int i = 0; i < queryFileNames.length; i++) {
			List<BaseQuery> bqs = QueryReader.readQueries(queryFileNames[i]);
			IndexSearcher is = iss[i];
			IndexReader ir = is.getIndexReader();

			String outputFileName = resDirNames[i] + "kld-fb_wv-exp.txt";

			TextFileWriter writer = new TextFileWriter(outputFileName);

			for (int j = 0; j < bqs.size(); j++) {
				BaseQuery bq = bqs.get(j);

				Indexer<String> wordIndexer = new Indexer<String>();
				Counter<String> qwcs1 = AnalyzerUtils.getWordCounts(bq.getSearchText(), analyzer);

				SparseVector qlm1 = VectorUtils.toSparseVector(qwcs1, wordIndexer, true);
				qlm1.normalize();

				Counter<String> qwcs2 = expand(ir, vSearcher, qwcs1);

				SparseVector qlm2 = VectorUtils.toSparseVector(qwcs2, wordIndexer, true);
				qlm2.normalize();

				int num_ret_docs = 1000;

				BooleanQuery lbq = AnalyzerUtils.getQuery(VectorUtils.toCounter(qlm1, wordIndexer));
				SparseVector docScores = SearcherUtils.search(lbq, is, num_ret_docs);

				WordCountBox wcb = WordCountBox.getWordCountBox(is, docScores, wordIndexer, CommonFieldNames.CONTENT);

				KLDivergenceScorer scorer = new KLDivergenceScorer();
				// scorer.score(wcb, qlm1);

				RelevanceModelBuilder rmb = new RelevanceModelBuilder();
				SparseVector rm = rmb.getRelevanceModel(wcb, docScores);
				// SparseVector prm = rmb.getPositionalRelevanceModel(qLM, wcb3,
				// docScores);

				double mixture = 0.5;

				SparseVector qlm3 = VectorMath.addAfterMultiply(qlm2, 1 - mixture, rm, mixture);

				docScores = scorer.score(wcb, qlm3);

				docScores.normalizeAfterSummation();

				System.out.println(bq);
				System.out.printf("QM1:\t%s\n", VectorUtils.toCounter(qlm1, wordIndexer));
				System.out.printf("QM2:\t%s\n", VectorUtils.toCounter(qlm2, wordIndexer));
				System.out.printf("QM3:\t%s\n", VectorUtils.toCounter(qlm3, wordIndexer));

				SearcherUtils.write(writer, bq.getId(), docScores);
			}

			writer.close();
		}
	}

	public void searchByKldFbWordVectorExp2() throws Exception {
		System.out.println("search by KLD FB Word Vector Exp 2.");

		Word2VecSearcher vSearcher = new Word2VecSearcher(
				Word2VecModel.fromSerFile("../../data/medical_ir/trec_cds/word2vec_model_stem.ser.gz"));

		for (int i = 0; i < queryFileNames.length; i++) {
			List<BaseQuery> bqs = QueryReader.readQueries(queryFileNames[i]);
			IndexSearcher is = iss[i];
			IndexReader ir = is.getIndexReader();

			String outputFileName = resDirNames[i] + "kld-fb_wv-exp_2.txt";

			TextFileWriter writer = new TextFileWriter(outputFileName);

			for (int j = 0; j < bqs.size(); j++) {
				BaseQuery bq = bqs.get(j);

				Indexer<String> wordIndexer = new Indexer<String>();
				Counter<String> qwcs1 = AnalyzerUtils.getWordCounts(bq.getSearchText(), analyzer);

				SparseVector qlm1 = VectorUtils.toSparseVector(qwcs1, wordIndexer, true);
				qlm1.normalize();

				SparseVector tfidfs = VectorUtils.toSparseVector(qwcs1, wordIndexer, true);

				for (int k = 0; k < tfidfs.size(); k++) {
					int w = tfidfs.indexAt(k);
					String word = wordIndexer.getObject(w);
					double doc_freq = ir.docFreq(new Term(CommonFieldNames.CONTENT, word));
					double tf = Math.log(tfidfs.valueAt(k)) + 1;
					double num_docs = ir.maxDoc();
					double idf = Math.log((num_docs + 1) / doc_freq);
					double tfidf = tf * idf;
					tfidfs.setAt(k, tfidf);
				}

				ArrayMath.unitVector(tfidfs.values(), tfidfs.values());

				double[][] wvs = vSearcher.getVectors();
				double[][] sims = ArrayMath.matrix(qlm1.size());
				double[] cents = ArrayUtils.copy(tfidfs.values());

				CounterMap cm = new CounterMap();

				for (int k = 0; k < qlm1.size(); k++) {
					int w1 = qlm1.indexAt(k);
					double tfidf1 = tfidfs.valueAt(k);
					String word1 = wordIndexer.getObject(w1);

					if (word1.startsWith("<N")) {
						continue;
					}

					double[] wv1 = wvs[w1];
					for (int l = k + 1; l < qlm1.size(); l++) {
						int w2 = qlm1.indexAt(l);
						double tfidf2 = tfidfs.valueAt(l);
						String word2 = wordIndexer.getObject(w2);

						if (word2.startsWith("<N")) {
							continue;
						}

						double[] wv2 = wvs[w2];
						double cosine = ArrayMath.dotProduct(wv1, wv2);
						double weight = cosine * tfidf1 * tfidf2;
						if (cosine > 0.7) {
							sims[k][l] = weight;
							sims[l][k] = weight;
							cm.setCount(word1, word2, weight);
						}
					}
				}

				// cm.normalize();
				// cm = cm.invert();

				ArrayMath.normalizeColumns(sims);

				ArrayMath.randomWalk(sims, cents, 10, 0.000001, 0.8);

				SparseVector qlm2 = qlm1.copy();
				qlm2.setValues(cents);

				// qlm2 = VectorMath.addAfterScale(new ohs.matrix.Vector[] {
				// qlm1, qlm2 }, new double[] { .5, .5 });

				int num_ret_docs = 1000;

				BooleanQuery lbq = AnalyzerUtils.getQuery(VectorUtils.toCounter(qlm1, wordIndexer));
				SparseVector docScores = SearcherUtils.search(lbq, is, num_ret_docs);

				WordCountBox wcb = WordCountBox.getWordCountBox(is, docScores, wordIndexer, CommonFieldNames.CONTENT);

				KLDivergenceScorer scorer = new KLDivergenceScorer();
				// scorer.score(wcb, qlm1);

				RelevanceModelBuilder rmb = new RelevanceModelBuilder();
				SparseVector rm = rmb.getRelevanceModel(wcb, docScores);
				// SparseVector prm = rmb.getPositionalRelevanceModel(qLM, wcb3,
				// docScores);

				double mixture = 0.5;

				SparseVector qlm3 = VectorMath.addAfterMultiply(qlm2, 1 - mixture, rm, mixture);

				docScores = scorer.score(wcb, qlm3);

				docScores.normalizeAfterSummation();

				System.out.println(bq);
				System.out.printf("QM1:\t%s\n", VectorUtils.toCounter(qlm1, wordIndexer));
				System.out.printf("QM2:\t%s\n", VectorUtils.toCounter(qlm2, wordIndexer));
				System.out.printf("QM3:\t%s\n", VectorUtils.toCounter(qlm3, wordIndexer));

				SearcherUtils.write(writer, bq.getId(), docScores);
			}

			writer.close();
		}

	}

	public void searchByKldFbWordVectorPrior() throws Exception {
		System.out.println("search by KLD FB Word Vector Prior.");

		Word2VecSearcher vSearcher = new Word2VecSearcher(
				Word2VecModel.fromSerFile("../../data/medical_ir/ohsumed/word2vec_model_stem.ser.gz"));
		for (int i = 0; i < queryFileNames.length; i++) {
			List<BaseQuery> bqs = QueryReader.readQueries(queryFileNames[i]);
			IndexSearcher is = iss[i];
			IndexReader ir = is.getIndexReader();

			String outputFileName = resDirNames[i] + "kld-fb_wv-priors.txt";

			// FileUtils.deleteFilesUnder(resDirNames[i]);

			TextFileWriter writer = new TextFileWriter(outputFileName);

			for (int j = 0; j < bqs.size(); j++) {
				BaseQuery bq = bqs.get(j);

				Indexer<String> wordIndexer = new Indexer<String>();
				Counter<String> qwcs = AnalyzerUtils.getWordCounts(bq.getSearchText(), analyzer);

				SparseVector qlm = VectorUtils.toSparseVector(qwcs, wordIndexer, true);
				qlm.normalize();

				SparseVector eqlm = qlm.copy();

				int num_ret_docs = 1000;

				BooleanQuery lbq = AnalyzerUtils.getQuery(VectorUtils.toCounter(eqlm, wordIndexer));
				SparseVector docScores = SearcherUtils.search(lbq, is, num_ret_docs);
				SparseVector docPriors = docScores.copy();

				WordCountBox wcb = WordCountBox.getWordCountBox(is, docScores, wordIndexer, CommonFieldNames.CONTENT);

				double[] qwv = getVectorSum(ir, vSearcher, qwcs);
				double[][] dwvs = new double[num_ret_docs][];

				for (int k = 0; k < num_ret_docs; k++) {
					int docId = docScores.indexAt(k);
					SparseVector dwcs = wcb.getDocToWordCounts().rowAt(k);
					double[] dwv = getVectorSum(ir, vSearcher, VectorUtils.toCounter(dwcs, wordIndexer));
					dwvs[k] = dwv;
				}

				double[][] sim_mat = ArrayMath.matrix(dwvs.length, 0);

				for (int k = 0; k < dwvs.length; k++) {
					double[] dwv1 = dwvs[k];
					sim_mat[k][k] = 1;

					for (int l = k + 1; l < dwvs.length; l++) {
						double[] dwv2 = dwvs[l];
						double cosine = ArrayMath.cosine(dwv1, dwv2);
						sim_mat[k][l] = cosine;
						sim_mat[l][k] = cosine;
					}
				}

				for (int k = 0; k < sim_mat.length; k++) {
					double[] sim = sim_mat[k];
					int[] indexes = ArrayUtils.rankedIndexes(sim);

					for (int l = 10; l < sim.length; l++) {
						sim[indexes[l]] = 0;
					}
				}

				LA.transpose(sim_mat);

				ArrayMath.normalizeColumns(sim_mat);

				double[] cents = new double[sim_mat.length];
				ArrayUtils.setAll(cents, 1f / cents.length);

				ArrayMath.randomWalk(sim_mat, cents, 10, 0.00000001, 0.85);
				docPriors.setValues(cents);
				docPriors.summation();

				RelevanceModelBuilder rmb = new RelevanceModelBuilder();
				SparseVector rm = rmb.getRelevanceModel(wcb, docScores, docPriors);

				double rm_mixture = 0.5;

				eqlm = VectorMath.addAfterMultiply(qlm, 1 - rm_mixture, rm, rm_mixture);

				KLDivergenceScorer scorer = new KLDivergenceScorer();
				docScores = scorer.score(wcb, eqlm);

				System.out.println(bq);
				System.out.printf("QM1:\t%s\n", VectorUtils.toCounter(qlm, wordIndexer));
				System.out.printf("QM2:\t%s\n", VectorUtils.toCounter(eqlm, wordIndexer));

				SearcherUtils.write(writer, bq.getId(), docScores);
			}

			writer.close();
		}
	}

	public void searchByKldFbWordVectors() throws Exception {
		System.out.println("search by KLD FB Word Vectors.");

		Word2VecSearcher vSearcher = new Word2VecSearcher(
				Word2VecModel.fromSerFile("../../data/medical_ir/ohsumed/word2vec_model_stem.ser.gz"));

		for (int i = 0; i < queryFileNames.length; i++) {
			List<BaseQuery> bqs = QueryReader.readQueries(queryFileNames[i]);
			IndexSearcher is = iss[i];
			IndexReader ir = is.getIndexReader();

			String outputFileName = resDirNames[i] + "kld-fb_wv.txt";

			TextFileWriter writer = new TextFileWriter(outputFileName);

			for (int j = 0; j < bqs.size(); j++) {
				BaseQuery bq = bqs.get(j);

				Indexer<String> wordIndexer = new Indexer<String>();
				Counter<String> qwcs = AnalyzerUtils.getWordCounts(bq.getSearchText(), analyzer);

				SparseVector qlm = VectorUtils.toSparseVector(qwcs, wordIndexer, true);
				qlm.normalize();

				SparseVector eqlm = qlm.copy();

				BooleanQuery lbq = AnalyzerUtils.getQuery(VectorUtils.toCounter(eqlm, wordIndexer));
				SparseVector docScores = SearcherUtils.search(lbq, is, 1000);

				WordCountBox wcb = WordCountBox.getWordCountBox(is, docScores, wordIndexer, CommonFieldNames.CONTENT);

				ParsimoniousLanguageModelEstimator e = new ParsimoniousLanguageModelEstimator(wcb);
				e.estimate();

				RelevanceModelBuilder rmb = new RelevanceModelBuilder();
				SparseVector rm = rmb.getRelevanceModel(wcb, docScores);
				// SparseVector prm = rmb.getPositionalRelevanceModel(qLM, wcb3,
				// docScores);

				double rm_mixture = 0.5;

				eqlm = VectorMath.addAfterMultiply(qlm, 1 - rm_mixture, rm, rm_mixture);

				KLDivergenceScorer scorer = new KLDivergenceScorer();
				docScores = scorer.score(wcb, eqlm);

				SparseVector docScores2 = docScores.copy();
				double[] qwv = getVectorSum(ir, vSearcher, qwcs);

				for (int k = 0; k < docScores.size(); k++) {
					int docId = docScores.indexAt(k);
					SparseVector dwcs = wcb.getDocToWordCounts().rowAt(k);
					double[] dwv = getVectorSum(ir, vSearcher, VectorUtils.toCounter(dwcs, wordIndexer));
					double cosine = ArrayMath.cosine(qwv, dwv);
					docScores2.setAt(k, cosine);
				}

				ArrayMath.multiply(docScores.values(), docScores2.values(), docScores.values());
				docScores.normalizeAfterSummation();

				System.out.println(bq);
				System.out.printf("QM1:\t%s\n", VectorUtils.toCounter(qlm, wordIndexer));
				System.out.printf("QM2:\t%s\n", VectorUtils.toCounter(eqlm, wordIndexer));

				SearcherUtils.write(writer, bq.getId(), docScores);
			}

			writer.close();
		}
	}

	public void searchByQLD() throws Exception {
		System.out.println("search by QLD.");

		for (int i = 0; i < queryFileNames.length; i++) {
			List<BaseQuery> bqs = QueryReader.readQueries(queryFileNames[i]);
			IndexSearcher is = iss[i];

			FileUtils.deleteFilesUnder(resDirNames[i]);

			String outFileName = resDirNames[i] + "qld.txt";

			TextFileWriter writer = new TextFileWriter(outFileName);

			for (int j = 0; j < bqs.size(); j++) {
				BaseQuery bq = bqs.get(j);
				BooleanQuery lbq = AnalyzerUtils.getQuery(bq.getSearchText(), analyzer);
				Map<Integer, String> docIdMap = Generics.newHashMap();
				SparseVector docScores = SearcherUtils.search(lbq, is, 1000, docIdMap);
				SearcherUtils.write(writer, bq.getId(), docScores, docIdMap);
			}
			writer.close();
		}
	}

	public void testKLDFBWiki() throws Exception {
		System.out.println("search by KLD FB.");
		IndexSearcher wis = SearcherUtils.getIndexSearcher(MIRPath.WIKI_INDEX_DIR);

		Set<String> stopPrefixes = WikiXmlDataHandler.getStopPrefixes();

		for (int i = 0; i < queryFileNames.length; i++) {
			List<BaseQuery> bqs = QueryReader.readQueries(queryFileNames[i]);
			IndexSearcher is = iss[i];
			IndexReader ir = is.getIndexReader();

			String outputFileName = MIRPath.OutputDirNames[i] + "wiki/titles.txt";

			TextFileWriter writer = new TextFileWriter(outputFileName);

			for (int j = 0; j < bqs.size(); j++) {
				BaseQuery bq = bqs.get(j);

				Indexer<String> wordIndexer = new Indexer<String>();
				Counter<String> qwcs = AnalyzerUtils.getWordCounts(bq.getSearchText(), analyzer);

				SparseVector qlm = VectorUtils.toSparseVector(qwcs, wordIndexer, true);
				qlm.normalize();

				SparseVector eqlm = qlm.copy();
				SparseVector docScores = null;

				BooleanQuery lbq = AnalyzerUtils.getQuery(VectorUtils.toCounter(eqlm, wordIndexer));
				docScores = SearcherUtils.search(lbq, wis, 50);

				docScores.sortValues();

				Counter<String> titleScores = Generics.newCounter();

				for (int k = 0; k < docScores.size(); k++) {
					int docid = docScores.indexAt(k);
					double score = docScores.valueAt(k);
					String title = wis.getIndexReader().document(docid).get(CommonFieldNames.TITLE);
					if (WikiXmlDataHandler.accept(stopPrefixes, title)) {
						titleScores.setCount(title, score);
					}
				}

				StringBuffer sb = new StringBuffer(bq.toString());
				sb.append("\nTitles:\n" + titleScores.toStringSortedByValues(true, true, titleScores.size(), " "));

				Counter<String> wwcs = Generics.newCounter();

				for (String title : titleScores.keySet()) {
					Counter<String> c = AnalyzerUtils.getWordCounts(title, analyzer);
					for (Entry<String, Double> e : c.entrySet()) {
						wwcs.incrementCount(e.getKey(), e.getValue());
					}
				}

				sb.append("\nQWCS:");

				int loc = 0;
				for (String word : qwcs.getSortedKeys()) {
					int cnt1 = (int) qwcs.getCount(word);
					int cnt2 = (int) wwcs.getCount(word);
					if (cnt2 > 0) {
						sb.append(String.format("\n%d\t%s\t%d\t%d", ++loc, word, cnt1, cnt2));
					}
				}
				sb.append("\n\n");

				writer.write(sb.toString());
			}

			writer.close();
		}
	}

}
