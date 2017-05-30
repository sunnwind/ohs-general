package ohs.ir.medical.general;

import ohs.corpus.dump.ClueWebDumper;

public class MIRPath {

	public static final String DATA_DIR = "../../data/medical_ir/";

	public static final String STOPWORD_MALLET_FILE = DATA_DIR + "stopword_mallet.txt";

	public static final String STOPWORD_INQUERY_FILE = DATA_DIR + "stopword_inquery.txt";

	public static final String EEM_LOG_FILE = DATA_DIR + "eem_log.txt";

	public static final String NEW_EEM_LOG_FILE = DATA_DIR + "new_eem_log.txt";

	public static final String PERFORMANCE_FILE = DATA_DIR + "performance.txt";

	public static final String PERFORMANCE_SUMMARY_FILE = DATA_DIR + "performance_summary.txt";

	public static final String PERFORMANCE_DETAIL_FILE = DATA_DIR + "performance_detail.txt";

	public static final String VOCAB_FILE = DATA_DIR + "vocab.txt.gz";

	/*
	 * BioASQ
	 */

	public static final String BIOASQ_DIR = DATA_DIR + "bioasq/";

	public static final String BIOASQ_COL_DIR = BIOASQ_DIR + "col/";

	public static final String BIOASQ_COL_RAW_DIR = BIOASQ_COL_DIR + "raw/";

	public static final String BIOASQ_COL_LINE_DIR = BIOASQ_COL_DIR + "line/";

	public static final String BIOASQ_COL_TOK_DIR = BIOASQ_COL_DIR + "tok/";

	public static final String BIOASQ_COL_DC_DIR = BIOASQ_COL_DIR + "dc/";

	public static final String BIOASQ_COL_INDEX_DIR = BIOASQ_COL_DIR + "idx/";

	public static final String BIOASQ_MESH_TREE_SER_FILE = BIOASQ_DIR + "mesh_tree.ser.gz";

	public static final String BIOASQ_MESH_RES_DIR = BIOASQ_DIR + "res/";

	public static final String BIOASQ_MESH_RES_SEARCH_DIR = BIOASQ_MESH_RES_DIR + "search/";

	/*
	 * MeSH
	 */

	public static final String MESH_DIR = DATA_DIR + "mesh/";

	public static final String MESH_COL_DIR = MESH_DIR + "col/";

	public static final String MESH_COL_RAW_DIR = MESH_COL_DIR + "raw/";

	public static final String MESH_COL_RAW_DESCRIPTOR_FILE = MESH_COL_RAW_DIR + "d2017.bin";

	public static final String MESH_COL_RAW_QUALIFIER_FILE = MESH_COL_RAW_DIR + "q2017.bin";

	public static final String MESH_COL_RAW_SUPPLEMENTARY_FILE = MESH_COL_RAW_DIR + "c2017.bin";

	public static final String MESH_COL_LINE_DIR = MESH_COL_DIR + "line/";

	public static final String MESH_COL_DC_DIR = MESH_COL_DIR + "dc/";

	public static final String MESH_COL_INDEX_DIR = MESH_COL_DIR + "idx/";

	/*
	 * UMLS
	 */

	public static final String UMLS_DIR = DATA_DIR + "umls/";

	public static final String UMLS_COL_DIR = UMLS_DIR + "col/";

	public static final String UMLS_COL_RAW_DIR = UMLS_COL_DIR + "raw/";

	/*
	 * Wikipedia
	 */

	public static final String WIKI_DIR = DATA_DIR + "wiki/";

	public static final String WIKI_COL_DIR = WIKI_DIR + "col/";

	public static final String WIKI_COL_XML_FILE = WIKI_COL_DIR + "raw/enwiki-20170201-pages-articles-multistream.xml.bz2";

	public static final String WIKI_COL_LINE_DIR = WIKI_COL_DIR + "line/";

	public static final String WIKI_COL_TOK_DIR = WIKI_COL_DIR + "tok/";

	public static final String WIKI_COL_DC_DIR = WIKI_COL_DIR + "dc/";

	public static final String WIKI_COL_INDEX_DIR = WIKI_COL_DIR + "idx/";

	public static final String WIKI_COL_COCNT_DIR = WIKI_COL_DIR + "cocnt/";

	public static final String WIKI_PAGE_FILE = WIKI_DIR + "wiki_page.csv.gz";

	public static final String WIKI_CATEGORY_FILE = WIKI_DIR + "wiki_category.csv.gz";

	public static final String WIKI_CATEGORYLINK_FILE = WIKI_DIR + "wiki_categorylink.csv.gz";

	public static final String WIKI_VOCAB_FILE = WIKI_DIR + "vocab.txt.gz";

	public static final String WIKI_INDEX_DIR = WIKI_DIR + "index/";

	public static final String WIKI_INDEX_SENT_DIR = WIKI_DIR + "index_sent/";

	public static final String WIKI_CATEGORY_INDEX_DIR = WIKI_DIR + "cat_index/";

	public static final String WIKI_REDIRECT_TITLE_FILE = WIKI_DIR + "redirects.txt";

	public static final String WIKI_TITLE_FILE = WIKI_DIR + "titles.txt";

	public static final String WIKI_DOC_ID_MAP_FILE = WIKI_DIR + "document_id_map.txt.gz";

	public static final String WIKI_CATEGORY_MAP_FILE = WIKI_DIR + "category_map.txt";

	public static final String WIKI_CATEGORY_COUNT_FILE = WIKI_DIR + "category_count.txt";

	public static final String WIKI_DOC_PRIOR_FILE = WIKI_DIR + "document_prior.ser";

	/*
	 * clueweb 2012
	 */

	public static final String CLUEWEB_DIR = DATA_DIR + "clueweb/";

	public static final String CLUEWEB_COL_DIR = CLUEWEB_DIR + "col/";

	public static final String CLUEWEB_COL_RAW_DIR = CLUEWEB_COL_DIR + "raw/";

	public static final String CLUEWEB_COL_LINE_DIR = CLUEWEB_COL_DIR + "line/";

	public static final String CLUEWEB_COL_TOK_DIR = CLUEWEB_COL_DIR + "tok/";

	public static final String CLUEWEB_COL_DC_DIR = CLUEWEB_COL_DIR + "dc/";

	public static final String CLUEWEB_COL_INDEX_DIR = CLUEWEB_COL_DIR + "idx/";

	public static final String CLUEWEB_FILE_NAME_FILE = CLUEWEB_DIR + "clueweb12_disk_b_file_names.txt";

	/*
	 * CLEF eHealth
	 */

	public static final String CLEF_EH_DIR = DATA_DIR + "clef_ehealth/";

	public static final String CLEF_EH_2014_DIR = CLEF_EH_DIR + "2014/";

	public static final String CLEF_EH_2015_DIR = CLEF_EH_DIR + "2015/";

	public static final String CLEF_EH_2016_DIR = CLEF_EH_DIR + "2016/";

	public static final String CLEF_EH_2017_DIR = CLEF_EH_DIR + "2017/";

	public static final String CLEF_EH_2014_COL_DIR = CLEF_EH_2014_DIR + "col/";

	public static final String CLEF_EH_2014_COL_RAW_DIR = CLEF_EH_2014_COL_DIR + "raw/";

	public static final String CLEF_EH_2014_COL_LINE_DIR = CLEF_EH_2014_COL_DIR + "line/";

	public static final String CLEF_EH_2014_COL_TOK_DIR = CLEF_EH_2014_COL_DIR + "tok/";

	public static final String CLEF_EH_2014_COL_DC_DIR = CLEF_EH_2014_COL_DIR + "dc/";

	public static final String CLEF_EH_2014_COL_INDEX_DIR = CLEF_EH_2014_COL_DIR + "idx/";

	public static final String CLEF_EH_2014_VALID_DOC_NO_FILE = CLEF_EH_2014_DIR + "cds-docnos.txt";

	public static final String CLEF_EH_2014_QUERY_DIR = CLEF_EH_2014_DIR + "query/";

	public static final String CLEF_EH_2013_QUERY_FILE = CLEF_EH_2014_QUERY_DIR + "queries.clef2013ehealth.1-50.test.xml";

	public static final String CLEF_EH_2014_QUERY_FILE = CLEF_EH_2014_QUERY_DIR + "queries.clef2014ehealth.1-50.test.en.xml";

	public static final String CLEF_EH_2014_REL_JUDGE_FILE = CLEF_EH_2014_QUERY_DIR + "clef2014t3.qrels.test.graded.txt";

	public static final String CLEF_EH_2015_QUERY_DIR = CLEF_EH_2015_DIR + "query/";

	public static final String CLEF_EH_2015_QUERY_FILE = CLEF_EH_2015_QUERY_DIR + "clef2015.test.queries-EN.txt";

	public static final String CLEF_EH_2015_REL_JUDGE_FILE = CLEF_EH_2015_QUERY_DIR + "qrels.clef2015.test.graded.txt";

	public static final String CLEF_EH_2016_QUERY_DIR = CLEF_EH_2016_DIR + "query/";

	public static final String CLEF_EH_2016_QUERY_FILE = CLEF_EH_2016_QUERY_DIR + "queries2016.xml";

	public static final String CLEF_EH_2016_REL_JUDGE_FILE = CLEF_EH_2016_QUERY_DIR + "task1.qrels";

	/*
	 * TREC CDS
	 */

	public static final String TREC_CDS_DIR = DATA_DIR + "trec_cds/";

	public static final String TREC_CDS_2014_DIR = TREC_CDS_DIR + "2014/";

	public static final String TREC_CDS_2014_COL_DIR = TREC_CDS_2014_DIR + "col/";

	public static final String TREC_CDS_2014_COL_RAW_DIR = TREC_CDS_2014_COL_DIR + "raw/";

	public static final String TREC_CDS_2014_COL_LINE_DIR = TREC_CDS_2014_COL_DIR + "line/";

	public static final String TREC_CDS_2014_COL_TOK_DIR = TREC_CDS_2014_COL_DIR + "tok/";

	public static final String TREC_CDS_2014_COL_DC_DIR = TREC_CDS_2014_COL_DIR + "dc/";

	public static final String TREC_CDS_2014_COL_INDEX_DIR = TREC_CDS_2014_COL_DIR + "idx/";

	public static final String TREC_CDS_2014_QUERY_DIR = TREC_CDS_2014_DIR + "query/";

	public static final String TREC_CDS_2014_DUPLICATION_FILE_1 = TREC_CDS_2014_DIR + "collection/duplicates-1.txt";

	public static final String TREC_CDS_2014_DUPLICATION_FILE_2 = TREC_CDS_2014_DIR + "collection/duplicates-2.txt";

	public static final String TREC_CDS_2014_VALID_DOC_ID_FILE = TREC_CDS_2014_DIR + "cds-docnos.txt";

	public static final String TREC_CDS_2014_QUERY_FILE = TREC_CDS_2014_QUERY_DIR + "topics-2014.xml";

	public static final String TREC_CDS_2014_REL_JUDGE_FILE = TREC_CDS_2014_QUERY_DIR + "qrels-treceval-2014.txt";

	public static final String TREC_CDS_2015_DIR = TREC_CDS_DIR + "2015/";

	public static final String TREC_CDS_2015_QUERY_DIR = TREC_CDS_2015_DIR + "query/";

	public static final String TREC_CDS_2015_QUERY_A_FILE = TREC_CDS_2015_QUERY_DIR + "topics-2015-A.xml";

	public static final String TREC_CDS_2015_QUERY_B_FILE = TREC_CDS_2015_QUERY_DIR + "topics-2015-B.xml";

	public static final String TREC_CDS_2015_REL_JUDGE_FILE = TREC_CDS_2015_QUERY_DIR + "qrels-treceval-2015.txt";

	public static final String TREC_CDS_2016_DIR = TREC_CDS_DIR + "2016/";

	public static final String TREC_CDS_2016_COL_DIR = TREC_CDS_2016_DIR + "col/";

	public static final String TREC_CDS_2016_COL_RAW_DIR = TREC_CDS_2016_COL_DIR + "raw/";

	public static final String TREC_CDS_2016_COL_LINE_DIR = TREC_CDS_2016_COL_DIR + "line/";

	public static final String TREC_CDS_2016_COL_TOK_DIR = TREC_CDS_2016_COL_DIR + "tok/";

	public static final String TREC_CDS_2016_COL_DC_DIR = TREC_CDS_2016_COL_DIR + "dc/";

	public static final String TREC_CDS_2016_COL_INDEX_DIR = TREC_CDS_2016_COL_DIR + "idx/";

	public static final String TREC_CDS_2016_QUERY_DIR = TREC_CDS_2016_DIR + "query/";

	public static final String TREC_CDS_2016_QUERY_FILE = TREC_CDS_2016_QUERY_DIR + "topics2016.xml";

	public static final String TREC_CDS_2016_REL_JUDGE_FILE = TREC_CDS_2016_QUERY_DIR + "qrels-treceval-2016.txt";

	/*
	 * TREC Genomics
	 */

	public static final String TREC_GENOMICS_DIR = DATA_DIR + "trec_genomics/";

	public static final String TREC_GENO_2007_DIR = TREC_GENOMICS_DIR + "2007/";

	public static final String TREC_GENO_2007_COL_DIR = TREC_GENO_2007_DIR + "col/";

	public static final String TREC_GENO_2007_COL_RAW_DIR = TREC_GENO_2007_COL_DIR + "raw/";

	public static final String TREC_GENO_2007_COL_LINE_DIR = TREC_GENO_2007_COL_DIR + "line/";

	public static final String TREC_GENO_2007_COL_TOK_DIR = TREC_GENO_2007_COL_DIR + "tok/";

	public static final String TREC_GENO_2007_COL_DC_DIR = TREC_GENO_2007_COL_DIR + "dc/";

	public static final String TREC_GENO_2007_COL_INDEX_DIR = TREC_GENO_2007_COL_DIR + "idx/";

	public static final String TREC_GENO_2007_QUERY_DIR = TREC_GENO_2007_DIR + "query/";

	public static final String TREC_GENO_2007_QUERY_FILE = TREC_GENO_2007_QUERY_DIR + "2007topics.txt";

	public static final String TREC_GENO_2007_RELEVANCE_JUDGE_FILE = TREC_GENO_2007_QUERY_DIR + "trecgen2007.all.judgments.tsv.txt";

	/*
	 * TREC Precision Medicine
	 */

	public static final String TREC_PM_DIR = DATA_DIR + "trec_pm/";

	public static final String TREC_PM_2017_DIR = TREC_PM_DIR + "2017/";

	public static final String TREC_PM_2017_COL_MEDLINE_DIR = TREC_PM_2017_DIR + "col_medline/";

	public static final String TREC_PM_2017_COL_CLINICAL_DIR = TREC_PM_2017_DIR + "col_clinical/";

	public static final String TREC_PM_2017_COL_MEDLINE_RAW_DIR = TREC_PM_2017_COL_MEDLINE_DIR + "raw/";

	public static final String TREC_PM_2017_COL_MEDLINE_LINE_DIR = TREC_PM_2017_COL_MEDLINE_DIR + "line/";

	public static final String TREC_PM_2017_COL_MEDLINE_TOK_DIR = TREC_PM_2017_COL_MEDLINE_DIR + "tok/";

	public static final String TREC_PM_2017_COL_MEDLINE_DC_DIR = TREC_PM_2017_COL_MEDLINE_DIR + "dc/";

	public static final String TREC_PM_2017_COL_MEDLINE_INDEX_DIR = TREC_PM_2017_COL_MEDLINE_DIR + "idx/";

	public static final String TREC_PM_2017_COL_CLINICAL_RAW_DIR = TREC_PM_2017_COL_CLINICAL_DIR + "raw/";

	public static final String TREC_PM_2017_COL_CLINICAL_LINE_DIR = TREC_PM_2017_COL_CLINICAL_DIR + "line/";

	public static final String TREC_PM_2017_COL_CLINICAL_TOK_DIR = TREC_PM_2017_COL_CLINICAL_DIR + "tok/";

	public static final String TREC_PM_2017_COL_CLINICAL_DC_DIR = TREC_PM_2017_COL_CLINICAL_DIR + "dc/";

	public static final String TREC_PM_2017_COL_CLINICAL_INDEX_DIR = TREC_PM_2017_COL_CLINICAL_DIR + "idx/";

	public static final String TREC_PM_2017_QUERY_DIR = TREC_PM_2017_DIR + "query/";

	public static final String TREC_PM_2017_QUERY_FILE = TREC_PM_2017_QUERY_DIR + "2017topics.txt";

	public static final String TREC_PM_2017_RELEVANCE_JUDGE_FILE = TREC_PM_2017_QUERY_DIR + "trecgen2017.all.judgments.tsv.txt";

	/*
	 * OHSUMED
	 */

	public static final String OHSUMED_DIR = DATA_DIR + "ohsumed/";

	public static final String OHSUMED_COL_DIR = OHSUMED_DIR + "col/";

	public static final String OHSUMED_COL_RAW_DIR = OHSUMED_COL_DIR + "raw/";

	public static final String OHSUMED_COL_LINE_DIR = OHSUMED_COL_DIR + "line/";

	public static final String OHSUMED_COL_TOK_DIR = OHSUMED_COL_DIR + "tok/";

	public static final String OHSUMED_COL_DC_DIR = OHSUMED_COL_DIR + "dc/";

	public static final String OHSUMED_COL_INDEX_DIR = OHSUMED_COL_DIR + "idx/";

	public static final String OHSUMED_QUERY_DIR = OHSUMED_DIR + "query/";

	public static final String OHSUMED_QUERY_FILE = OHSUMED_QUERY_DIR + "queries.txt";

	public static final String OHSUMED_RELEVANCE_JUDGE_FILE = OHSUMED_QUERY_DIR + "judged.txt";

	/*
	 * SNOMED CT
	 */

	public static final String SNOMED_DIR = DATA_DIR + "snomed_ct/";

	/*
	 * Common File Name Sets
	 */

	public static String[] DataDirNames = {

			TREC_CDS_2014_DIR, TREC_CDS_2014_DIR,

			OHSUMED_DIR, TREC_GENO_2007_DIR,

			CLEF_EH_2014_DIR, CLEF_EH_2014_DIR };

	public static String[] QueryFileNames = {

			TREC_CDS_2014_QUERY_FILE, TREC_CDS_2015_QUERY_A_FILE,

			OHSUMED_QUERY_FILE, TREC_GENO_2007_QUERY_FILE,

			CLEF_EH_2014_QUERY_FILE, CLEF_EH_2015_QUERY_FILE };

	public static final String[] RelevanceFileNames = {

			TREC_CDS_2014_REL_JUDGE_FILE, TREC_CDS_2015_REL_JUDGE_FILE,

			OHSUMED_RELEVANCE_JUDGE_FILE, TREC_GENO_2007_RELEVANCE_JUDGE_FILE,

			CLEF_EH_2014_REL_JUDGE_FILE, CLEF_EH_2015_REL_JUDGE_FILE };

	public static final String[] TaskDirNames = {

			TREC_CDS_2014_DIR, TREC_CDS_2015_DIR,

			OHSUMED_DIR, TREC_GENO_2007_DIR,

			CLEF_EH_2014_DIR, CLEF_EH_2015_DIR };

	public static final String[] IndexDirNames = addSuffix(DataDirNames, "index/");

	public static String[] ResultDirNames = addSuffix(TaskDirNames, "res/");

	public static final String[] OutputDirNames = addSuffix(ResultDirNames, "out/");

	public static final String[] DocIdMapFileNames = addSuffix(DataDirNames, "document_id_map.txt.gz");

	private static String[] addSuffix(String[] s, String suffix) {
		String[] ret = new String[s.length];
		for (int i = 0; i < s.length; i++) {
			ret[i] = s[i] + suffix;
		}
		return ret;
	}

}
