# BSD 3-Clause License
#
# Copyright (c) 2022, ZHENG Leon
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import math
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time

from src import utils


class ButterflyFact(object):
    def __init__(self, factors, matrix, shared=True):
        self.size = 2**factors
        self.support = utils.support_DFT(factors)
        self.permutation = [
            torch.from_numpy(perm).cdouble() for perm in utils.perm_DFT(factors)
        ]
        self.shared = shared
        self.logits = [
            (torch.randn(1) * 0.00001).double().requires_grad_()
            for i in range(3 * (factors - 1))
        ]
        self.sharedlogits = [
            (torch.randn(8)).double().requires_grad_() for i in range(factors - 1)
        ]
        self.A = torch.from_numpy(matrix).cdouble()
        self.X = []
        self.Y = [torch.from_numpy(sp).double() for sp in self.support]
        for sp in self.support:
            w = torch.randn(sp.shape[0], sp.shape[1], dtype=torch.cdouble) * math.sqrt(
                0.5
            )
            self.X.append(w.requires_grad_())

    def training_sharedlogits(self, lr, num_iter=1000, optimize="SGD"):
        loss_fn = nn.MSELoss()

        loss_array = []
        logit_array = []
        best = np.inf

        # Choose optimizer
        if optimize == "Adam":
            optimizer = optim.Adam(self.X + self.sharedlogits, lr=lr)
        elif optimize == "SGDMomentum":
            optimizer = optim.SGD(self.X + self.sharedlogits, lr=lr, momentum=0.9)
        elif optimize == "SGD":
            optimizer = optim.SGD(self.X + self.sharedlogits, lr=lr)

        allP = []
        for i in range(len(self.X) - 1):
            currP = torch.randn(8, self.size, self.size).cdouble()
            Pc = [torch.eye(self.size).cdouble(), self.permutation[3 * i]]
            Pb = [torch.eye(self.size).cdouble(), self.permutation[3 * i + 1]]
            Pa = [torch.eye(self.size).cdouble(), self.permutation[3 * i + 2]]
            for i1 in range(2):
                for i2 in range(2):
                    for i3 in range(2):
                        currP[i1 * 4 + i2 * 2 + i3] = torch.mm(
                            torch.mm(Pc[i1], Pb[i2]), Pa[i3]
                        )
            allP.append(currP)

        for index in range(num_iter):
            optimizer.zero_grad()
            Z = self.X[0] * self.Y[0]

            for i in range(len(self.X) - 1):
                Z = torch.mm(Z, self.X[i + 1] * self.Y[i + 1])

            for i in range(len(self.X) - 1):
                Z = torch.mm(
                    Z,
                    torch.einsum(
                        "i, ijk->jk",
                        [
                            nn.functional.softmax(self.sharedlogits[i]).cdouble(),
                            allP[i],
                        ],
                    ),
                )
            loz = self.logits[0].data.numpy()
            logit_array.append(loz)

            loss = 0.5 * torch.sum(((Z - self.A) * (Z - self.A).conj()).real)
            loss.backward()
            print("Iteration ", index, ": ", loss)

            loss_array.append(loss.data.numpy())
            best = min(best, loss_array[-1])
            optimizer.step()

    def training_nosharedlogits(self, lr, num_iter=1000, optimize="SGD"):
        loss_fn = nn.MSELoss()

        loss_array = []
        logit_array = []
        best = np.inf

        # Choose optimizer
        if optimize == "Adam":
            optimizer = optim.Adam(self.X + self.logits, lr=lr)
        elif optimize == "SGDMomentum":
            optimizer = optim.SGD(self.X + self.logits, lr=lr, momentum=0.9)
        elif optimize == "SGD":
            optimizer = optim.SGD(self.X + self.logits, lr=lr)

        for index in range(num_iter):
            optimizer.zero_grad()
            Z = self.X[0] * self.Y[0]

            for i in range(len(self.X) - 1):
                Z = torch.mm(Z, self.X[i + 1] * self.Y[i + 1])
            if not self.shared:
                for i in range(len(self.logits)):
                    Z = torch.mm(
                        Z,
                        (
                            torch.eye(self.size) * torch.sigmoid(self.logits[i])
                            + self.permutation[i] * (1 - torch.sigmoid(self.logits[i]))
                        ).type(torch.cdouble),
                    )
            else:
                for i in range(len(self.X) - 1):
                    for j in range(3):
                        Z = torch.mm(
                            Z,
                            (
                                torch.eye(self.size) * torch.sigmoid(self.logits[j])
                                + self.permutation[i * 3 + j]
                                * (1 - torch.sigmoid(self.logits[j]))
                            ).type(torch.cdouble),
                        )
            loss = 0.5 * torch.sum(((Z - self.A) * (Z - self.A).conj()).real)
            loss.backward()
            print("Iteration ", index, ": ", loss)

            loss_array.append(loss.data.numpy())
            best = min(best, loss_array[-1])
            optimizer.step()


class ButterflyFact_BuiltinComplex(object):
    def __init__(self, factors, matrix, shared=True):
        self.size = 2**factors
        self.support = utils.support_DFT(factors)
        self.permutation = [
            torch.from_numpy(perm).double() for perm in utils.perm_DFT(factors)
        ]
        self.shared = shared

        if not shared:
            self.logits = [
                (torch.randn(1) * 0.00001).double().requires_grad_()
                for i in range(3 * (factors - 1))
            ]
        else:
            self.logits = [
                (torch.randn(1) * 0.00001).double().requires_grad_() for i in range(3)
            ]

        self.A_real = torch.from_numpy(matrix.real).double()
        self.A_imag = torch.from_numpy(matrix.imag).double()
        self.X_real = []
        self.X_imag = []

        self.Y = [torch.from_numpy(sp).double() for sp in self.support]
        for sp in self.support:
            w_real = torch.randn(
                sp.shape[0], sp.shape[1], dtype=torch.double
            ) * math.sqrt(0.5)
            w_imag = torch.randn(
                sp.shape[0], sp.shape[1], dtype=torch.double
            ) * math.sqrt(0.5)
            self.X_real.append(w_real.requires_grad_())
            self.X_imag.append(w_imag.requires_grad_())

    def training(
        self,
        lr,
        num_iter=50,
        optimize="Adam",
        stopping_time=[49, 99, 149, 199],
        verbal=False,
        time_record=False,
    ):
        loss_fn = nn.MSELoss()

        loss_array = []
        running_time = []
        accuracy = []
        best = np.inf

        # Choose optimizer
        if optimize == "Adam":
            optimizer = optim.Adam(self.X_real + self.X_imag + self.logits, lr=lr)
        elif optimize == "SGDMomentum":
            optimizer = optim.SGD(
                self.X_real + self.X_imag + self.logits, lr=lr, momentum=0.9
            )
        elif optimize == "SGD":
            optimizer = optim.SGD(self.X_real + self.X_imag + self.logits, lr=lr)

        begin = time.time()
        refine_time = 0

        running_time_per_iter = []
        loss_per_iter = []
        for index in range(num_iter):
            begin_record = time.time()

            optimizer.zero_grad()
            Z_real = self.X_real[0] * self.Y[0]
            Z_imag = self.X_imag[0] * self.Y[0]

            for i in range(len(self.X_real) - 1):
                T_real = Z_real * 1.0
                T_imag = Z_imag * 1.0
                Z_real = torch.mm(
                    T_real, self.X_real[i + 1] * self.Y[i + 1]
                ) - torch.mm(T_imag, self.X_imag[i + 1] * self.Y[i + 1])
                Z_imag = torch.mm(
                    T_imag, self.X_real[i + 1] * self.Y[i + 1]
                ) + torch.mm(T_real, self.X_imag[i + 1] * self.Y[i + 1])

            if not self.shared:
                for i in range(len(self.logits)):
                    Z_real = torch.mm(
                        Z_real,
                        torch.eye(self.size) * torch.sigmoid(self.logits[i])
                        + self.permutation[i] * (1 - torch.sigmoid(self.logits[i])),
                    )
                    Z_imag = torch.mm(
                        Z_imag,
                        torch.eye(self.size) * torch.sigmoid(self.logits[i])
                        + self.permutation[i] * (1 - torch.sigmoid(self.logits[i])),
                    )
            else:
                for i in range(len(self.X_real) - 1):
                    for j in range(3):
                        Z_real = torch.mm(
                            Z_real,
                            torch.eye(self.size) * torch.sigmoid(self.logits[j])
                            + self.permutation[i * 3 + j]
                            * (1 - torch.sigmoid(self.logits[j])),
                        )
                        Z_imag = torch.mm(
                            Z_imag,
                            torch.eye(self.size) * torch.sigmoid(self.logits[j])
                            + self.permutation[i * 3 + j]
                            * (1 - torch.sigmoid(self.logits[j])),
                        )

            loss = 0.5 * torch.sum((Z_real - self.A_real) ** 2) + 0.5 * torch.sum(
                (Z_imag - self.A_imag) ** 2
            )
            loss.backward()
            if verbal:
                print("Iteration ", index, ": ", loss)

            loss_array.append(loss.data.numpy())
            best = min(best, loss_array[-1])
            optimizer.step()

            end_record = time.time()
            running_time_per_iter.append(end_record - begin_record)

            if index in stopping_time:
                refine_begin = time.time()
                if not time_record:
                    accuracy.append(self.refining())
                else:
                    acc, refine_loss, refine_temps = self.refining(time_record=True)
                    loss_array = loss_array + refine_loss
                    running_time_per_iter = running_time_per_iter + refine_temps

                refine_end = time.time()
                end = time.time()
                running_time.append(end - begin - refine_time)
                refine_time += refine_end - refine_begin

        return running_time, accuracy, loss_array, running_time_per_iter

    def refining(self, num_iter=20, time_record=False):
        X_real = [
            (torch.from_numpy(x.data.numpy())).double().requires_grad_()
            for x in self.X_real
        ]
        X_imag = [
            (torch.from_numpy(x.data.numpy())).double().requires_grad_()
            for x in self.X_imag
        ]
        logits = []
        loss_array = []
        running_time = []

        for x in self.logits:
            if x.data.numpy()[0] > 0:
                logits.append(1)
            else:
                logits.append(0)

        def closure():
            optimizer.zero_grad()
            Z_real = X_real[0] * self.Y[0]
            Z_imag = X_imag[0] * self.Y[0]

            for i in range(len(self.X_real) - 1):
                T_real = Z_real * 1.0
                T_imag = Z_imag * 1.0
                Z_real = torch.mm(T_real, X_real[i + 1] * self.Y[i + 1]) - torch.mm(
                    T_imag, X_imag[i + 1] * self.Y[i + 1]
                )
                Z_imag = torch.mm(T_imag, X_real[i + 1] * self.Y[i + 1]) + torch.mm(
                    T_real, X_imag[i + 1] * self.Y[i + 1]
                )

            if not self.shared:
                for i in range(len(self.logits)):
                    Z_real = torch.mm(
                        Z_real,
                        torch.eye(self.size) * logits[i]
                        + self.permutation[i] * (1 - logits[i]),
                    )
                    Z_imag = torch.mm(
                        Z_imag,
                        torch.eye(self.size) * logits[i]
                        + self.permutation[i] * (1 - logits[i]),
                    )
            else:
                for i in range(len(self.X_real) - 1):
                    for j in range(3):
                        Z_real = torch.mm(
                            Z_real,
                            torch.eye(self.size) * logits[j]
                            + self.permutation[i * 3 + j] * (1 - logits[j]),
                        )
                        Z_imag = torch.mm(
                            Z_imag,
                            torch.eye(self.size) * logits[j]
                            + self.permutation[i * 3 + j] * (1 - logits[j]),
                        )
            loss = 0.5 * torch.sum((Z_real - self.A_real) ** 2) + 0.5 * torch.sum(
                (Z_imag - self.A_imag) ** 2
            )
            loss.backward()
            return loss

        best = np.inf
        optimizer = optim.LBFGS(X_real + X_imag, line_search_fn="strong_wolfe")
        for index in range(num_iter):
            begin = time.time()
            optimizer.step(closure)
            end = time.time()
            running_time.append(end - begin)
            loss_array.append(closure().data.numpy())
        if not time_record:
            return closure().data.numpy()
        else:
            return closure().data.numpy(), loss_array, running_time
