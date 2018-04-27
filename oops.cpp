/* Pneumonoultramicroscopicsilicovolcanoconiosis */
#include <stdio.h>
#include <cmath>
#include <string>
#include <vector>
#include <list>
#include <valarray>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <chrono>
#define MAXTHREADS 8
#define ADAM_BETA1 0.9
#define ADAM_BETA2 0.999
#define ADAM_EPSILON 1.0e-8
#define PREDICTDAYS 5
#define TARGETINDEX 3
#if defined(_OPENMP)
#include <omp.h>
#else
inline int omp_get_thread_num() { return 0; }
inline void omp_set_num_threads(int n) { return; }
#endif
typedef std::valarray<double> Vec;
typedef std::vector<Vec> Mat;

inline double square(double&& x) {
	return x * x;
}

std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
std::uniform_real_distribution<double> distribution(-0.05, 0.05);
void randomInitialization(Vec& target) {
	std::generate(std::begin(target), std::end(target), [&] () { return distribution(generator); });
	return;
}

struct ArgParse {
	double regularizer = 0.001;
	double adamlr = 0.001;
	int batchSize = 1024;
	int dimension = 128;/* individual stock embedding */
	int ompThread = 1;
	int history = 60;/* default 3 months */
	int features = 5;
	int epoch = 500;
	bool testMode = true;
	bool openmp = false;
	std::string savePrefix = std::string("./oops_");
	std::string inputFile;
	std::string setting = "";
	
	ArgParse() {}
	~ArgParse() {}
	
	bool parseArgs(int argc, char** argv) {
		if (argc < 2) {
			printf(">>> The program needs at least 1 argument: InputFile\n");
			return false;
		}
		inputFile = std::string(argv[1]);
		if (parseOptions(argc, argv, 2)) {
			#if defined(_OPENMP)
			ompThread = std::min(ompThread, MAXTHREADS);
			ompThread = std::max(1, ompThread);
			omp_set_num_threads(ompThread);
			openmp = true;
			#else
			ompThread = 1;
			#endif
			setting += "InputFile         : " + inputFile + "\n";
			setting += "EmbeddingDimension: " + std::to_string(dimension) + "\n";
			setting += "UseHistory        : " + std::to_string(history) + "\n";
			setting += "FeatureLength     : " + std::to_string(features) + "\n";
			setting += "TrainingEpoches   : " + std::to_string(epoch) + "\n";
			setting += "BatchSize         : " + std::to_string(batchSize) + "\n";
			setting += "LearningRate      : " + std::to_string(adamlr) + "\n";
			setting += "Regularizer       : " + std::to_string(regularizer) + "\n";
			setting += "TestMode          : " + std::string(testMode ? "Yes" : "No") + "\n";
			setting += "OpenmpThreads     : " + (openmp ? std::to_string(ompThread) : "disabled") + "\n";
			setting += "SavePrefix        : " + savePrefix + "\n";
			setting += "==================================================\n";
			printf("%s", setting.c_str());
			return true;
		}
		printf(">>> arguments error\n");
		return false;
	}
	
	bool parseOptions(int argc, char** argv, int optStart) {
		for (int i = optStart; i < argc; ++i) {
			if (std::string(argv[i]) == "--omp") {
				if (++i >= argc)	return false;
				ompThread = std::stoi(std::string(argv[i]));
			} else if (std::string(argv[i]) == "-l") {
				if (++i >= argc)	return false;
				double temp = std::stod(std::string(argv[i]));
				adamlr = (temp < 0.0) ? adamlr : temp;
			} else if (std::string(argv[i]) == "-r") {
				if (++i >= argc)	return false;
				double temp = std::stod(std::string(argv[i]));
				regularizer = (temp < 0.0) ? regularizer : temp;
			} else if (std::string(argv[i]) == "-d") {
				if (++i >= argc)	return false;
				int temp = std::stoi(std::string(argv[i]));
				dimension = (temp < 1) ? dimension : temp;
			} else if (std::string(argv[i]) == "-h") {
				if (++i >= argc)	return false;
				int temp = std::stoi(std::string(argv[i]));
				history = (temp < 1) ? history : temp;
			} else if (std::string(argv[i]) == "-f") {
				if (++i >= argc)	return false;
				int temp = std::stoi(std::string(argv[i]));
				features = (temp < 0) ? features : temp;
			} else if (std::string(argv[i]) == "-e") {
				if (++i >= argc)	return false;
				int temp = std::stoi(std::string(argv[i]));
				epoch = (temp < 0) ? epoch : temp;
			} else if (std::string(argv[i]) == "-b") {
				if (++i >= argc)	return false;
				int temp = std::stoi(std::string(argv[i]));
				batchSize = (temp < 64) ? batchSize : temp;
			} else if (std::string(argv[i]) == "-nt") {
				testMode = false;
			} else if (std::string(argv[i]) == "-s") {
				if (++i >= argc)	return false;
				savePrefix = std::string(argv[i]);
			} else {
				return false;
			}
		}
		return true;
	}
	
} args;

const double scoreWeight[] = { 0.0, 0.1, 0.15, 0.2, 0.25, 0.3 };

struct Stock {
	static int idCount;
	const int ID;/* index of row in Element */
	std::vector<Vec> history;
	std::vector<int> weekday;
	std::vector<int> month;
	std::vector<double> prediction;
	std::vector<long int> truePrice;
	Vec dataAvg = Vec(0.0, args.features);
	Vec dataStd = Vec(0.0, args.features);
	int firstPredictIndex = 0;
	
	Stock(): ID(idCount++) {
		history.reserve(1600);
		weekday.reserve(1600);
		month.reserve(1600);
	}
	~Stock() {}
	
	void add(int& _year, int& _month, int& _weekday, Vec& val) {
		val = val.apply([] (const double& v) { return std::round(v * 100.0); });
		dataAvg += val;
		dataStd += val * val;
		history.push_back(val);
		weekday.push_back(_weekday);
		month.push_back(_month);
		return;
	}
	
	Stock* init() {
		weekday.insert(weekday.end(), { 1, 2, 3, 4, 5 });
		month.insert(month.end(), 5, month.back());
		firstPredictIndex = static_cast<int>(history.size()) - (args.testMode ? PREDICTDAYS : 0);
		for (auto it = history.begin() + firstPredictIndex - 1; it != history.end(); ++it)
			truePrice.push_back(std::lrint((*it)[TARGETINDEX]));
		/* normalization */
		double&& dataSizeDouble = static_cast<double>(history.size());
		dataAvg = dataAvg / dataSizeDouble;
		dataStd = std::sqrt(dataStd / dataSizeDouble - dataAvg * dataAvg);
		for (Vec& v: history)
			v = (v - dataAvg) / dataStd;
		return this;
	}
	
	std::vector<long int> rescale() {
		std::vector<long int> predPrice({ truePrice[0] });
		for (double& d: prediction)
			predPrice.push_back(std::lrint(d * dataStd[TARGETINDEX] + dataAvg[TARGETINDEX]));
		return predPrice;
	}
	
	void predictOutput(FILE* sp) {
		std::vector<long int> predPrice = rescale();
		for (size_t i = 1; i <= PREDICTDAYS; ++i) {
			long int preddiff = predPrice[i] - predPrice[i - 1];
			int    predchange = (preddiff == 0) ? 0 : ((preddiff > 0) ? +1 : -1);
			fprintf(sp, ",%d,%.2lf", predchange, static_cast<double>(predPrice[i]) / 100.0);
		}
		return;
	}
	
	double evaluate(FILE* sp) {
		std::vector<long int> predPrice = rescale();
		double totalScore = 0.0;
		fprintf(sp, "p\t%ld\tt\t%ld\n", predPrice[0], truePrice[0]);
		for (size_t i = 1; i <= PREDICTDAYS; ++i) {
			long int preddiff = predPrice[i] - predPrice[i - 1];
			int    predchange = (preddiff == 0) ? 0 : ((preddiff > 0) ? +1 : -1);
			long int truediff = truePrice[i] - truePrice[i - 1];
			int    truechange = (truediff == 0) ? 0 : ((truediff > 0) ? +1 : -1);
			long int&&  error = std::abs(predPrice[i] - truePrice[i]);
			double&&    score = scoreWeight[i] * (((predchange == truechange) ? 0.5 : 0.0) +
				(static_cast<double>(truePrice[i] - error) / static_cast<double>(truePrice[i])) * 0.5);
			totalScore += score;
			fprintf(sp, "%d\t%ld\t%d\t%ld\t%.2lf\n", predchange, predPrice[i], truechange, truePrice[i], score);
		}
		return totalScore;
	}
	
};
int Stock::idCount = 0;

class Element {
private:
	static std::vector<Element*> collection;
	const int nodeCount;/* set to 1 as scalar */
	const int dimension;
	Mat gradients[MAXTHREADS];
	Mat adamBias1;
	Mat adamBias2;
	Mat embedding;
	int adam_t = 0;/* learning times */
	
	void update(double batchSizeDouble) {/* gradient descent */
		const double&& rho = args.adamlr * ((++adam_t > 30000) ? 1.0 :
			(std::sqrt(1.0 - std::pow(ADAM_BETA2, adam_t)) / (1.0 - std::pow(ADAM_BETA1, adam_t))));
		for (int i = 0; i < nodeCount; ++i) {
			Vec g(2.0 * args.regularizer * embedding[i]);
			for (int t = 0; t < args.ompThread; ++t) {
				g += gradients[t][i];
				gradients[t][i] = 0.0;
			}
			g /= batchSizeDouble;
			adamBias1[i] = ADAM_BETA1 * adamBias1[i] + (1.0 - ADAM_BETA1) * g;
			adamBias2[i] = ADAM_BETA2 * adamBias2[i] + (1.0 - ADAM_BETA2) * g * g;
			embedding[i] -= rho * (adamBias1[i] / (std::sqrt(adamBias2[i]) + ADAM_EPSILON));
		}
		return;
	}
	
public:
	Element(int c, int d): nodeCount(c), dimension(d) {
		Vec zeros(0.0, dimension);
		for (int t = 0; t < args.ompThread; ++t)
			gradients[t].assign(nodeCount, zeros);
		adamBias1.assign(nodeCount, zeros);
		adamBias2.assign(nodeCount, zeros);
		embedding.assign(nodeCount, zeros);
		for (Vec& x: embedding)
			randomInitialization(x);
		collection.push_back(this);
	}
	
	Vec& operator[] (int index) {
		return embedding[index];
	}
	
	void setGradient(Vec g, int entry = 0) {
		gradients[omp_get_thread_num()][entry] += g;
		return;
	}
	
	/* Wx + b */
	Vec computeWxpb(Vec& x, Vec& b) {
		Vec result(b.size());
		std::transform(embedding.begin(), embedding.end(), std::begin(b), std::begin(result),
			[&x] (const Vec& w, const double& bi) { return (w * x).sum() + bi; });
		return result;
	}
	
	/* the derivative of || Wx + b - y || ^ 2 wrt W is 2 * (Wx + b - y) * x.T */
	void setGradient(Vec& g, Vec& x) {
		const int&& tid = omp_get_thread_num();
		std::transform(gradients[tid].begin(), gradients[tid].end(), std::begin(g), gradients[tid].begin(),
			[&x] (Vec& a, double& gi) { return a + x * gi; });
		return;
	}
	
	/* the derivative of || Wx + b - y || ^ 2 wrt x is 2 * W.T * (Wx + b - y) */
	Vec computeGradient(Vec& g, const std::slice& targetSlice) {
		Vec targetGradient(0.0, targetSlice.size());
		auto git = std::begin(g);
		for (const Vec& w: embedding) {
			targetGradient += Vec(w[targetSlice]) * *git;
			++git;
		}
		return targetGradient;
	}
	
	void dump(std::string fileName) {
		FILE* fp = fopen(fileName.c_str(), "wb");
		fwrite(static_cast<const void*>(&nodeCount), sizeof nodeCount, 1, fp);
		fwrite(static_cast<const void*>(&dimension), sizeof dimension, 1, fp);
		for (Vec& e: embedding)
			fwrite(static_cast<const void*>(&*std::begin(e)), sizeof(double), e.size(), fp);
		fclose(fp);
		return;
	}
	
	static void updateAll(int batchSize) {
		for (Element* e: collection)
			e->update(static_cast<double>(batchSize));
		return;
	}
	
};
std::vector<Element*> Element::collection;

int main(int argc, char** argv) {
	if (not args.parseArgs(argc, argv))
		return 0;
	
	std::unordered_map<std::string, Stock> stockMap;
	{
		char id[32];
		int year, month, weekday;
		FILE* fp = fopen(args.inputFile.c_str(), "r");
		while (fscanf(fp, "%s %d %d %d ", id, &year, &month, &weekday) == 4) {
			Vec feature(args.features);
			for (int i = 0; i < args.features; ++i)
				fscanf(fp, "%lf", &feature[i]);
			stockMap[std::string(id)].add(year, month, weekday, feature);
		}
		fclose(fp);
	}
	std::vector<std::pair<Stock*, int>> learningPair;
	for (auto& p: stockMap) {
		Stock* s = p.second.init();
		/* use [t - args.history, t - 1] to predict [t] */;
		for (int i = s->firstPredictIndex - 1; i >= args.history; --i)
			learningPair.push_back(std::make_pair(s, i));
	}
	learningPair.shrink_to_fit();
	const int&& learningSize = static_cast<int>(learningPair.size());
	
	const int learningCount = 3;
	Element   stockEmbedding(Stock::idCount, args.dimension);
	Element weekdayEmbedding(7, args.dimension);
	Element   monthEmbedding(12, args.dimension);
	Element weightMatrix(args.features, args.history * args.features + args.dimension * learningCount);
	Element   biasVector(1, args.features);
	
	const int xDimension = args.dimension * learningCount + args.history * args.features;
	const std::slice   stockSlice(                 0, args.dimension, 1);
	const std::slice weekdaySlice(args.dimension * 1, args.dimension, 1);
	const std::slice   monthSlice(args.dimension * 2, args.dimension, 1);
	const std::slice featureSlice(0, learningCount *  args.dimension, 1);
	std::vector<std::slice> historySlice(args.history + 1);
	/* latest history starting with smaller index */
	for (int i = 1; i <= args.history; ++i)
		historySlice[i] = std::slice(args.dimension * learningCount + (i-1) * args.features, args.features, 1);
	
	double totalTrainingTime = 0.0;
	for (int e = 1; e <= args.epoch; ++e) {
		std::random_shuffle(learningPair.begin(), learningPair.end());
		const auto ttime1 = std::chrono::high_resolution_clock::now();
		double epochLoss = 0.0;
		
		for (int currZ = 0; currZ < learningSize; ) {
			const int nextZ = std::min(learningSize, currZ + args.batchSize);
			#if defined(_OPENMP)
			#pragma omp parallel for reduction(+ : epochLoss)
			#endif
			for (int z = currZ; z < nextZ; ++z) {
				Stock* const currStock = learningPair[z].first;
				const int t = learningPair[z].second;
				
				Vec& y = currStock->history[t];
				Vec  x(xDimension);
				x[  stockSlice] =   stockEmbedding[currStock->ID];
				x[weekdaySlice] = weekdayEmbedding[currStock->weekday[t]];
				x[  monthSlice] =   monthEmbedding[currStock->month[t]];
				for (int i = 1; i <= args.history; ++i)
					x[historySlice[i]] = currStock->history[t - i];
				
				/* common gradient part */
				Vec Wxpbmy2 = (weightMatrix.computeWxpb(x, biasVector[0]) - y) * 2.0;
				epochLoss += square(Wxpbmy2.sum() / 2.0);
				
				biasVector.setGradient(Wxpbmy2);
				weightMatrix.setGradient(Wxpbmy2, x);
				Vec xGradient = weightMatrix.computeGradient(Wxpbmy2, featureSlice);
				stockEmbedding.setGradient(Vec(xGradient[stockSlice]), currStock->ID);
				weekdayEmbedding.setGradient(Vec(xGradient[weekdaySlice]), currStock->weekday[t]);
				monthEmbedding.setGradient(Vec(xGradient[monthSlice]), currStock->month[t]);
			}
			Element::updateAll(nextZ - currZ);
			currZ = nextZ;
		}
		
		const auto ttime2 = std::chrono::high_resolution_clock::now();
		double&& dt = std::chrono::duration<double>(ttime2 - ttime1).count();
		totalTrainingTime += dt;
		printf("\rEpoch %3d/%d - %.3fs - Loss: %.4e", e, args.epoch, dt, epochLoss);
		fflush(stdout);
	}
	printf("\nTotalTrainingTime: %.3fs\n", totalTrainingTime);
	
	biasVector.dump(args.savePrefix + "biasVector.txt");
	weightMatrix.dump(args.savePrefix + "weightMatrix.txt");
	stockEmbedding.dump(args.savePrefix + "stockEmbedding.txt");
	weekdayEmbedding.dump(args.savePrefix + "weekdayEmbedding.txt");
	monthEmbedding.dump(args.savePrefix + "monthEmbedding.txt");
	FILE* fp = fopen((args.savePrefix + "info.txt").c_str(), "w");
	fprintf(fp, "%s", args.setting.c_str());
	for (auto& kv: stockMap)
		fprintf(fp, "%s %d\n", kv.first.c_str(), kv.second.ID);
	fclose(fp);
	
	/* predict */
	FILE* sp = fopen((args.savePrefix + "output.txt").c_str(), "w");
	double totalScore = 0.0;
	for (auto& kv: stockMap) {
		fprintf(sp, "\n%s%s", kv.first.c_str(), args.testMode ? "\n" : "");
		Stock* const currStock = &kv.second;
		const int t = currStock->firstPredictIndex;
		
		std::list<Vec> historyInput;
		for (int i = 1; i <= args.history; ++i)
			historyInput.push_back(currStock->history[t - i]);
		
		for (int p = 0; p < PREDICTDAYS; ++p) {
			Vec x(xDimension);
			x[  stockSlice] =   stockEmbedding[currStock->ID];
			x[weekdaySlice] = weekdayEmbedding[currStock->weekday[t+p]];
			x[  monthSlice] =   monthEmbedding[currStock->month[t+p]];
			size_t i = 0;
			for (Vec& vec: historyInput)
				x[historySlice[++i]] = vec;
			Vec yp = weightMatrix.computeWxpb(x, biasVector[0]);
			historyInput.pop_back();
			historyInput.push_front(yp);
			currStock->prediction.push_back(yp[TARGETINDEX]);
		}
		if (args.testMode)
			totalScore += currStock->evaluate(sp);
		else
			currStock->predictOutput(sp);
	}
	fprintf(sp, args.testMode ? ("\nTotalScore: " + std::to_string(totalScore)).c_str() : "\n");
	fclose(sp);
	printf("\nTotalScore: %lf\n", totalScore);
	return 0;
}
