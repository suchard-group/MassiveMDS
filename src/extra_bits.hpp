#if 0

    template <size_t N> struct uint_{ };

    template <size_t N, typename Lambda, typename IterT>
    inline void unroller(const Lambda& f, const IterT& iter, uint_<N>) {
        unroller(f, iter, uint_<N-1>());
        f(iter + N);
    }

    template <typename Lambda, typename IterT>
    inline void unroller(const Lambda& f, const IterT& iter, uint_<0>) {
        f(iter);
    }

	template <typename Integer, typename Function>
	inline void for_each(Integer begin, Integer end, Function function) {
	    for (; begin != end; begin += 4) {
	        function(begin + 0);
	        function(begin + 1);
	        function(begin + 2);
	        function(begin + 3);
	    }
	}

	template <size_t UnrollFact, typename Integer, typename Function>
	inline void for_each_auto(Integer begin, Integer end, Function function) {
	    for (; begin != end; begin += UnrollFact) {
	        unroller(function, begin,  uint_<UnrollFact-1>());
	    }
	}
#else

#endif
