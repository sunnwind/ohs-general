package ohs.ir.medical.general;

import java.util.HashMap;
import java.util.Map;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.util.BytesRef;

import ohs.io.TextFileWriter;
import ohs.ir.lucene.common.CommonFieldNames;
import ohs.math.ArrayMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.utils.Timer;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class DocumentPriorEstimator {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		String[] indexDirNames = MIRPath.IndexDirNames;

		String[] docPriorFileNames = MIRPath.DocPriorFileNames;

		IndexSearcher[] indexSearchers = new IndexSearcher[indexDirNames.length];

		for (int i = 0; i < indexDirNames.length; i++) {
			indexSearchers[i] = SearcherUtils.getIndexSearcher(indexDirNames[i]);
		}

		DocumentPriorEstimator ds = new DocumentPriorEstimator(indexSearchers, docPriorFileNames);
		ds.estimate();

		System.out.println("process ends.");
	}

	private IndexSearcher[] iss;

	private String[] docPriorFileNames;

	private int num_colls;

	private double mixture_for_all_colls = 0;

	private double dirichlet_prior = 1500;

	public DocumentPriorEstimator(IndexSearcher[] iss, String[] docPriorFileNames) {
		super();
		this.iss = iss;
		this.docPriorFileNames = docPriorFileNames;
		num_colls = iss.length;
	}

	private void estimate() throws Exception {
		double[] cnt_sum_in_each_coll = new double[num_colls];
		double[] num_docs_in_each_coll = new double[num_colls];
		double cnt_sum_in_all_colls = 0;

		for (int i = 0; i < num_colls; i++) {
			Counter<Integer> counter = new Counter<Integer>();
			IndexReader ir = iss[i].getIndexReader();
			cnt_sum_in_each_coll[i] = ir.getSumTotalTermFreq(CommonFieldNames.CONTENT);
			num_docs_in_each_coll[i] = ir.maxDoc();
			cnt_sum_in_all_colls += cnt_sum_in_each_coll[i];
		}

		Timer timer = new Timer();
		timer.start();

		for (int i = 0; i < num_colls; i++) {
			IndexReader ir = iss[i].getIndexReader();
			TextFileWriter writer = new TextFileWriter(docPriorFileNames[i].replace("ser", "txt"));

			DenseVector docPriors = new DenseVector(ir.maxDoc());

			for (int j = 0; j < ir.maxDoc(); j++) {
				if ((j + 1) % 10000 == 0) {
					System.out.printf("\r%dth coll [%d/%d docs, %s]", i + 1, j + 1, (int) num_docs_in_each_coll[i], timer.stop());
				}
				Document doc = ir.document(j);

				Terms termVector = ir.getTermVector(j, CommonFieldNames.CONTENT);

				if (termVector == null) {
					continue;
				}

				Indexer<String> wordIndexer = new Indexer<String>();
				SparseVector docWordCounts = null;
				SparseVector[] collWordCountData = new SparseVector[num_colls];

				{
					TermsEnum termsEnum = null;
					termsEnum = termVector.iterator();

					BytesRef bytesRef = null;
					PostingsEnum postingsEnum = null;
					Counter<Integer> wcs = new Counter<Integer>();
					Map<Integer, Integer> locWords = new HashMap<Integer, Integer>();

					while ((bytesRef = termsEnum.next()) != null) {
						postingsEnum = termsEnum.postings(postingsEnum, PostingsEnum.ALL);

						if (postingsEnum.nextDoc() != 0) {
							throw new AssertionError();
						}

						String word = bytesRef.utf8ToString();
						// if (word.startsWith("<N") && word.endsWith(">")) {
						// continue;
						// }
						if (word.contains("<N")) {
							continue;
						}

						int w = wordIndexer.getIndex(word);
						int freq = postingsEnum.freq();
						wcs.incrementCount(w, freq);

						for (int k = 0; k < freq; k++) {
							final int position = postingsEnum.nextPosition();
							locWords.put(position, w);
						}
					}

					docWordCounts = VectorUtils.toSparseVector(wcs);
				}

				for (int k = 0; k < num_colls; k++) {
					Counter<Integer> counter = new Counter<Integer>();

					for (int w = 0; w < wordIndexer.size(); w++) {
						String word = wordIndexer.getObject(w);
						Term termInstance = new Term(CommonFieldNames.CONTENT, word);
						double count = iss[k].getIndexReader().totalTermFreq(termInstance);
						counter.setCount(w, count);
					}
					collWordCountData[k] = VectorUtils.toSparseVector(counter);
				}

				double sum_log_probs = 0;

				for (int k = 0; k < docWordCounts.size(); k++) {
					int w = docWordCounts.indexAt(k);
					double cnt_w_in_doc = docWordCounts.valueAt(k);
					String word = wordIndexer.getObject(w);

					double[] cnt_w_in_each_coll = new double[num_colls];
					double cnt_w_in_all_colls = 0;

					for (int u = 0; u < num_colls; u++) {
						cnt_w_in_each_coll[u] = collWordCountData[u].value(w);
						cnt_w_in_all_colls += cnt_w_in_each_coll[u];
					}

					double cnt_w_in_coll = cnt_w_in_each_coll[i];
					double cnt_sum_in_coll = cnt_sum_in_each_coll[i];

					double prob_w_in_coll = cnt_w_in_coll / cnt_sum_in_coll;
					double prob_w_in_all_colls = cnt_w_in_all_colls / cnt_sum_in_all_colls;

					double cnt_sum_in_doc = docWordCounts.sum();
					double prob_w_in_doc = (cnt_w_in_doc + dirichlet_prior * prob_w_in_coll) / (cnt_sum_in_doc + dirichlet_prior);
					prob_w_in_doc = (1 - mixture_for_all_colls) * prob_w_in_doc + mixture_for_all_colls * prob_w_in_all_colls;

					if (prob_w_in_doc == 0) {
						System.out.println();
					}

					double log_prob_w = Math.log(prob_w_in_doc);
					sum_log_probs += log_prob_w;
				}

				docPriors.set(j, sum_log_probs);
				writer.write(j + "\t" + sum_log_probs + "\n");

				// System.out.println(VectorUtils.toCounter(docWordCounts,
				// wordIndexer));
				// System.out.println(VectorUtils.toCounter(docWordProbs,
				// wordIndexer));
				// System.out.println();

				// writer.write(j + "\t" + doc_prior + "\n");
			}

			System.out.printf("\r%dth coll [%d/%d docs, %s]\n\n", i + 1, (int) num_docs_in_each_coll[i], (int) num_docs_in_each_coll[i],
					timer.stop());
			writer.close();

			double[] log_probs = docPriors.values();
			double log_prob_sum = ArrayMath.sumLogProbs(log_probs);

			ArrayMath.add(log_probs, -log_prob_sum, log_probs);
			ArrayMath.exp(log_probs, log_probs);

			docPriors.normalizeAfterSummation();

			docPriors.writeObject(docPriorFileNames[i]);

		}
	}
}
