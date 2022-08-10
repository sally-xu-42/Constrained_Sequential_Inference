from rayuela.base.automaton import Automaton
from rayuela.base.semiring import Boolean, Semiring
from collections import defaultdict as dd
from typing import Tuple

from rayuela.fsa.state import MinimizeState, State
from tqdm import tqdm


# Deterministic FSA
class DFSA(Automaton):
    # Construct from fsa
    def __init__(self, fsa):
        # if fsa is None:
        #     self.Q = {}
        #     self.initial_state = self.final_state = None
        #     self.delta = dd(lambda: {})
        #     self.Sigma = {}
        #     return
        
        from rayuela.fsa.fsa import FSA
        assert isinstance(fsa, FSA)
        assert fsa.deterministic
        assert fsa.R is Boolean
        # self.fsa = fsa  # save it for intersection

        state_map = {q: i for i, q in enumerate(fsa.Q)}  # from fsa state to dfsa

        self.Q: set = set(state_map.values())

        self.initial_state = [state_map[q] for q in fsa.Q if fsa.λ[q].score]
        assert len(self.initial_state) == 1
        self.initial_state = self.initial_state[0]

        self.final_state = [state_map[q] for q in fsa.Q if fsa.ρ[q].score]
        assert len(self.final_state) == 1
        self.final_state = self.final_state[0]

        # delta[i][a] = j
        self.delta = dd(lambda: {})

        self.Sigma = {a.sym for a in fsa.Sigma}

        # Build deterministic transition arcs
        for i in fsa.Q:
            for a, j, w in fsa.arcs(i):
                if w.score:
                    self.delta[state_map[i]][a.sym] = state_map[j]

    def accept(self, tokens) -> bool:
        """ determines whether a string is in the language """
        assert isinstance(tokens, list)
        cur = self.initial_state
        for a in tokens:
            nxt = self.delta[cur].get(a)
            if nxt is not None:
                cur = nxt
            else:  # no arc
                return False

        return cur == self.final_state
    
    def get_start(self) -> int:
        # print(f'FSA getting start {self.initial_state}')
        return self.initial_state

    def get_valid_actions(self, state: int, stack: int) -> list:
        actions = list(self.delta[state])
        # print(f'FSA get valid actions with state {state}, stack {stack} and actions {actions}')
        return actions
    
    def step(self, state: int, stack: int, action) -> Tuple[int, int]:  # returns to state, to stack
        # print(f'FSA step with state {state}, stack {stack} and action {action}')
        nxt = self.delta[state].get(action)
        if nxt is None:
            import dill
            dill.dump(self, open('bad_dfsa.dill', 'wb'))
            raise ValueError(f'Bad dfsa step with state {state}, stack {stack} and action {action}')
        return nxt, stack

    @property
    def num_states(self):
        return len(self.Q)

    def minimize(self):
        # Homework 5: Question 3
        print(f'Minimizing a fsa with {self.num_states} states...')
        from rayuela.base.partitions import PartitionRefinement

        # Assume deterministic and trim
        final_s = frozenset([self.final_state])
        non_final_s = frozenset(self.Q - final_s)
        P_cal = {final_s, non_final_s}
        for a in tqdm(self.Sigma):
            f_a = {}
            Q = set()
            for q in self.Q:
                p = self.delta[q].get(a)
                if p is not None:
                    f_a[q] = p
                    Q.add(q)
                else:
                    f_a[q] = q
            P_cal = PartitionRefinement(f_a, Q).hopcroft(P_cal)
        
        return self._block_fsa_construction(P_cal)

    def _block_fsa_construction(self, P_cal):
        # block fsa construction
        from rayuela.fsa.fsa import FSA
        mfsa = FSA()
        μ = {q: State(i) for i, Q in enumerate(P_cal) for q in Q}
        for i in self.Q:
            for a, j in self.delta[i].items():
                MinimizeState
                mfsa.add_arc(μ[i], a, μ[j])

        mfsa.set_I(μ[self.initial_state])
        mfsa.set_F(μ[self.final_state])
        return mfsa

    def _repr_html_(self):
        """
        When returned from a Jupyter cell, this will generate the FST visualization
        Based on: https://github.com/matthewfl/openfst-wrapper
        """
        from uuid import uuid4
        import json
        from collections import defaultdict
        ret = []
        if self.num_states == 0:
            return '<code>Empty FST</code>'

        if self.num_states > 64:
            return f'FST too large to draw graphic, use fst.ascii_visualize()<br /><code>FST(num_states={self.num_states})</code>'

        # print initial
        for q in [self.initial_state]:
            if q == self.final_state:
                label = f'{q}'
                color = 'af8dc3'
            else:
                label = f'{q}'
                color = '66c2a5'

            ret.append(
                f'g.setNode("{q}", {{ label: {json.dumps(label)} , shape: "circle" }});\n')
                # f'g.setNode("{repr(q)}", {{ label: {json.dumps(hash(label) // 1e8)} , shape: "circle" }});\n')

            ret.append(f'g.node("{q}").style = "fill: #{color}"; \n')

        # print normal
        for q in self.Q - {self.initial_state, self.final_state}:

            label = f'{q}'

            ret.append(
                f'g.setNode("{q}", {{ label: {json.dumps(label)} , shape: "circle" }});\n')
                # f'g.setNode("{repr(q)}", {{ label: {json.dumps(hash(label) // 1e8)} , shape: "circle" }});\n')
            ret.append(f'g.node("{q}").style = "fill: #8da0cb"; \n')

        # print final
        for q in [self.final_state]:
            # already added
            if q == self.initial_state:
                continue

            label = f'{q}'

            ret.append(
                f'g.setNode("{q}", {{ label: {json.dumps(label)} , shape: "circle" }});\n')
                # f'g.setNode("{repr(q)}", {{ label: {json.dumps(hash(label) // 1e8)} , shape: "circle" }});\n')
            ret.append(f'g.node("{q}").style = "fill: #fc8d62"; \n')

        for q in self.Q:
            to = defaultdict(list)
            for a, j in self.delta[q].items():
                label = f'{a}'
                to[j].append(label)

            for dest, values in to.items():
                if len(values) > 4:
                    values = values[0:3] + ['. . .']
                label = '\n'.join(values)
                ret.append(
                    f'g.setEdge("{q}", "{dest}", {{ arrowhead: "vee", label: {json.dumps(label)} }});\n')

        # if the machine is too big, do not attempt to make the web browser display it
        # otherwise it ends up crashing and stuff...
        if len(ret) > 256:
            return f'FST too large to draw graphic, use fst.ascii_visualize()<br /><code>FST(num_states={self.num_states})</code>'

        ret2 = ['''
        <script>
        try {
        require.config({
        paths: {
        "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3",
        "dagreD3": "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min"
        }
        });
        } catch {
            ["https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3.js",
            "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min.js"].forEach(function (src) {
            var tag = document.createElement('script');
            tag.src = src;
            document.body.appendChild(tag);
            })
        }
        try {
        requirejs(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        try {
        require(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        </script>
        <style>
        .node rect,
        .node circle,
        .node ellipse {
        stroke: #333;
        fill: #fff;
        stroke-width: 1px;
        }

        .edgePath path {
        stroke: #333;
        fill: #333;
        stroke-width: 1.5px;
        }
        </style>
        ''']

        obj = 'fst_' + uuid4().hex
        ret2.append(
            f'<center><svg width="850" height="600" id="{obj}"><g/></svg></center>')
        ret2.append('''
        <script>
        (function render_d3() {
        var d3, dagreD3;
        try { // requirejs is broken on external domains
            d3 = require('d3');
            dagreD3 = require('dagreD3');
        } catch (e) {
            // for google colab
            if(typeof window.d3 !== "undefined" && typeof window.dagreD3 !== "undefined") {
            d3 = window.d3;
            dagreD3 = window.dagreD3;
            } else { // not loaded yet, so wait and try again
            setTimeout(render_d3, 50);
            return;
            }
        }
        //alert("loaded");
        var g = new dagreD3.graphlib.Graph().setGraph({ 'rankdir': 'LR' });
        ''')
        ret2.append(''.join(ret))

        ret2.append(f'var svg = d3.select("#{obj}"); \n')
        ret2.append(f'''
        var inner = svg.select("g");

        // Set up zoom support
        var zoom = d3.zoom().scaleExtent([0.3, 5]).on("zoom", function() {{
        inner.attr("transform", d3.event.transform);
        }});
        svg.call(zoom);

        // Create the renderer
        var render = new dagreD3.render();

        // Run the renderer. This is what draws the final graph.
        render(inner, g);

        // Center the graph
        var initialScale = 0.75;
        svg.call(zoom.transform, d3.zoomIdentity.translate(
            (svg.attr("width") - g.graph().width * initialScale) / 2, 20).scale(initialScale));

        svg.attr('height', g.graph().height * initialScale + 50);
        }})();

        </script>
        ''')

        return ''.join(ret2)
