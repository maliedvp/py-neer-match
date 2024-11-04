"""
Reasoning module.

This module provides reasoning functionality for the entity matching tasks using logic
tensor networks.
"""

from neer_match.axiom_generator import AxiomGenerator
from neer_match.data_generator import DataGenerator
from neer_match.matching_model import NSMatchingModel
import tensorflow as tf
import ltn


class RefutationModel(NSMatchingModel):
    """A logic tensor network refutation model for entity matching tasks."""

    def __make_claims(self, axiom_generator, refutation):
        instructions = axiom_generator.data_generator.similarity_map.instructions
        rassoc = refutation[0]
        rindex = list(instructions.keys()).index(rassoc)
        rsim = refutation[1]
        if len(rsim) == 0:
            sindices = range(len(instructions[rassoc]))
        else:
            sindices = [instructions[rassoc].index(sim) for sim in rsim]

        id_predicate = ltn.Predicate.Lambda(lambda x: x)
        is_positive = ltn.Predicate.Lambda(lambda x: x > 0.0)
        r_predicate = ltn.Predicate(self.record_pair_network.field_networks[rindex])

        @tf.function
        def claims(features, labels):
            propositions = []

            y = ltn.Variable("y", labels)
            z = ltn.Variable("z", features[rassoc])
            x = [
                ltn.Variable(f"x{sindex}", features[rassoc][:, sindex])
                for sindex in sindices
            ]
            stmt = axiom_generator.ForAll(
                [x[0], y, z],
                axiom_generator.Implies(r_predicate(z), id_predicate(x[0])),
                mask=is_positive(y),
            )
            for v in x[1:]:
                stmt = axiom_generator.Or(
                    stmt,
                    axiom_generator.ForAll(
                        [v, y, z],
                        axiom_generator.Implies(r_predicate(z), id_predicate(v)),
                        mask=is_positive(y),
                    ),
                )
            propositions.append(stmt)

            kb = axiom_generator.FormAgg(propositions)
            sat = kb.tensor
            return sat

        return claims

    def __make_loss(self, q, claims, claim_sat_weight, axioms, beta, alpha):
        @tf.function
        def loss_clb(features, labels):
            preds = self.record_pair_network(features)
            bce = self.bce(labels, preds)
            asat = axioms(features, labels)
            csat = claims(features, labels)
            loss = (
                (1.0 - claim_sat_weight) * (1.0 - bce)
                + claim_sat_weight * csat
                + tf.keras.activations.elu(beta * (q - asat), alpha=alpha)
            )
            logs = {"BCE": bce, "ASat": asat, "Sat": csat, "Predicted": preds}
            return loss, logs

        return loss_clb

    def _training_loop_log_header(self):
        headers = ["Epoch", "BCE", "Recall", "Precision", "F1", "CSat", "ASat"]
        return "| " + " | ".join([f"{x:<10}" for x in headers]) + " |"

    def _training_loop_log_row(self, epoch, logs):
        recall = logs["TP"] / (logs["TP"] + logs["FN"])
        precision = logs["TP"] / (logs["TP"] + logs["FP"])
        f1 = 2.0 * precision * recall / (precision + recall)
        values = [
            logs["BCE"].numpy(),
            recall,
            precision,
            f1,
            logs["Sat"].numpy(),
            logs["ASat"].numpy(),
        ]
        row = f"| {epoch:<10} | " + " | ".join([f"{x:<10.4f}" for x in values]) + " |"
        return row

    def fit(
        self,
        left,
        right,
        matches,
        epochs,
        refutation,
        satisfiability_threshold=0.95,
        axioms_non_sat_scale=1.0,
        axioms_sat_scale=0.1,
        claim_sat_weight=1.0,
        verbose=1,
        log_mod_n=1,
        **kwargs,
    ):
        """Fit the refutation model."""
        if not (isinstance(refutation, str) or isinstance(refutation, tuple)):
            raise ValueError(
                "Refutation must be a string or a tuple."
                f"Instead got type {type(refutation)}."
            )
        if isinstance(refutation, str) or len(refutation) == 1:
            refutation = (refutation[0], None)
        if refutation[0] not in self.similarity_map.instructions.keys():
            raise ValueError(f"Association {refutation[0]} is not in similarity map.")
        if refutation[1] is None:
            refutation = (refutation[0], tuple())
        elif isinstance(refutation[1], str):
            refutation[1] = (refutation[1], tuple())
        for similarity in refutation[1]:
            if similarity not in self.similarity_map.instructions[refutation[0]]:
                raise ValueError(
                    f"Similarity {similarity} is not in association {refutation[0]}."
                )
        if satisfiability_threshold < 0.0 or satisfiability_threshold > 1.0:
            raise ValueError("Satisfiability threshold must be between 0 and 1")
        if axioms_non_sat_scale < 0.0:
            raise ValueError("Non-satisfiability scale must be positive")
        if axioms_sat_scale < 0.0:
            raise ValueError("Satisfiability scale must be positive")
        if claim_sat_weight < 0.0 or claim_sat_weight > 1.0:
            raise ValueError("Satisfiability weight must be between 0 and 1")
        # The remaining arguments are checked by the parent class' fit method

        data_generator = DataGenerator(
            self.record_pair_network.similarity_map, left, right, matches, **kwargs
        )
        axiom_generator = AxiomGenerator(data_generator)

        axioms = self._make_axioms(axiom_generator)
        claims = self.__make_claims(axiom_generator, refutation)
        loss_clb = self.__make_loss(
            satisfiability_threshold,
            claims,
            claim_sat_weight,
            axioms,
            axioms_non_sat_scale,
            axioms_sat_scale,
        )

        trainable_variables = self.record_pair_network.trainable_variables
        super()._training_loop(
            data_generator, loss_clb, trainable_variables, epochs, verbose, log_mod_n
        )
