"""
ZazorLayer — Спектральный Осьминог (мотивно-голографическая реализация)
======================================================================
Непрерывный поток, цветовые гармоники, многошкальная временная иерархия.
Архитектура: симплициальный комплекс над базисом, лежандрова геометрия,
мотивный пучок, гейт через коразмерность пересечения, поток Риччи,
динамическое расширение базиса (синестезия).
Все параметры адаптивны — выводятся из геометрии активных слотов.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# ------------------------------------------------------------
# Адаптивная конфигурация (без магических чисел)
# ------------------------------------------------------------
class ZazorConfig:
    """
    Хранит только размерности и начальные масштабы.
    Все пороги и скорости вычисляются динамически.
    """
    def __init__(
        self,
        motive_dim: int = 64,          # размерность мотивного вектора (может расти)
        max_slots: int = 128,          # максимальное число слотов
        num_basal: int = 4,            # начальное число базальных вершин (0-симплексов)
        color_dim: int = 9,            # размерность спектра (2K+1)
        time_window: int = 32,         # глубина временного буфера (адаптивна)
        initial_temperature: float = 1.0,
    ):
        self.motive_dim = motive_dim
        self.max_slots = max_slots
        self.num_basal = num_basal
        self.color_dim = color_dim
        self.time_window = time_window
        self.initial_temperature = initial_temperature

# ------------------------------------------------------------
# Гейт Theta: смешивание через коразмерность пересечения
# ------------------------------------------------------------
class ThetaGate(nn.Module):
    """Гейт, управляемый индексом пересечения носителей (без внешнего trust)."""
    def __init__(self, dim: int):
        super().__init__()
        # Преобразование для контекста и памяти -> общий знаменатель
        self.align = nn.Linear(2 * dim, dim)

    def forward(self, ctx: torch.Tensor, mem: torch.Tensor,
                active_ctx: torch.Tensor, active_mem: torch.Tensor) -> torch.Tensor:
        """
        ctx, mem: (d,) мотивные векторы контекста и памяти.
        active_ctx, active_mem: (n,) индикаторы активности вершин (носители).
        Возвращает C0.
        """
        # Коразмерность пересечения носителей
        intersection = (active_ctx * active_mem).sum()
        max_active = max(active_ctx.sum(), active_mem.sum())
        # Если оба пустые (крайний случай), τ = 1
        tau = 1.0 - (intersection / max_active) if max_active > 0 else 1.0

        # Интерполяция: при tau=0 (полное пересечение) -> ctx, при tau=1 -> mem
        C0 = mem + (1.0 - tau) * (ctx - mem)
        return C0

# ------------------------------------------------------------
# Вспомогательные модули
# ------------------------------------------------------------
class Critic(nn.Module):
    """Предсказатель удовлетворённости."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 3, dim), nn.ReLU(), nn.Linear(dim, 1))
    def forward(self, anchor, target, C0):
        return self.net(torch.cat([anchor, target, C0], dim=-1)).squeeze(-1)

class ActionHead(nn.Module):
    """Формирует базальный аффект."""
    def __init__(self, dim, basal_dim):
        super().__init__()
        self.net = nn.Linear(dim * 2, basal_dim + dim)
    def forward(self, C0, target):
        return self.net(torch.cat([C0, target], dim=-1))

# ------------------------------------------------------------
# Основной слой
# ------------------------------------------------------------
class ZazorLayer(nn.Module):
    def __init__(self, config: ZazorConfig):
        super().__init__()
        self.cfg = config
        d = config.motive_dim
        cdim = config.color_dim
        n_basal = config.num_basal

        # ---- Базис: базальные вершины (0-симплексы) ----
        # Инициализируем ортогональными векторами на сфере
        basis_raw = torch.randn(n_basal, d)
        # Ортогонализация Грама-Шмидта для начальной разнесённости
        q, _ = torch.linalg.qr(basis_raw.T)
        self.basis = nn.Parameter(q.T)  # (n_basal, d), строки — ортонормированы
        self.num_basal = n_basal

        # ---- Слоты: барицентрические координаты и спектры (p) ----
        # alpha_logits определяют положение на симплексе (softmax)
        self.alpha_logits = nn.Parameter(torch.randn(config.max_slots, n_basal) * 0.1)
        # spectrum — ковектор (цвет), интерпретируется как p
        self.spectrum = nn.Parameter(torch.rand(config.max_slots, cdim) * 0.1)
        # Инициализируем базальные слоты как вершины симплекса
        with torch.no_grad():
            for i in range(min(n_basal, config.max_slots)):
                self.alpha_logits[i, i] = 5.0  # почти one-hot
                # Простой спектр: одна гармоника
                self.spectrum[i, 0] = 0.5
                if cdim > 1 and i*2+1 < cdim:
                    self.spectrum[i, 2*i+1] = 0.8
                if cdim > 2 and i*2+2 < cdim:
                    self.spectrum[i, 2*i+2] = 0.4

        # ---- Возраст слотов (для устаревания) ----
        self.register_buffer('age', torch.zeros(config.max_slots))

        # ---- Маски активности ----
        self.register_buffer('active_mask', torch.ones(config.max_slots, dtype=torch.bool))

        # ---- Временные масштабы (привязаны к размерности симплекса) ----
        # Размерность симплекса = (число значимых вершин - 1). Храним как веса.
        # Масштаб ~ exp(-dim) — быстрые для низкоразмерных.
        self.time_scales = nn.Parameter(torch.ones(config.max_slots))
        self.time_scales.requires_grad = False  # обновляются динамически

        # ---- Якорь и цель ----
        self.anchor = nn.Parameter(torch.zeros(d))
        self.target = nn.Parameter(torch.zeros(d))

        # ---- Сети ----
        self.gate = ThetaGate(d)
        self.critic = Critic(d)
        self.action = ActionHead(d, n_basal)  # basal_dim = n_basal (динамическое)

        # ---- Буфер истории (для временного контекста) ----
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('history', torch.zeros(config.time_window, d))
        # Важность фреймов (размерность симплекса)
        self.register_buffer('frame_salience', torch.zeros(config.time_window))

        # ---- Статистики для адаптивных порогов ----
        self.register_buffer('dist_ma', torch.tensor(0.5))       # среднее расстояние мотивов
        self.register_buffer('spec_disp_ma', torch.tensor(0.1))  # дисперсия спектров
        self.register_buffer('residual_ma', torch.tensor(0.0))   # средний остаток проекции
        self.register_buffer('sat_ma', torch.tensor(0.5))        # средняя удовлетворённость

    # ------------------------------------------------------------
    # Получение полного мотивного вектора слота
    # ------------------------------------------------------------
    def get_motive(self, alpha_logits: torch.Tensor) -> torch.Tensor:
        """alpha_logits -> барицентрические координаты -> мотивный вектор."""
        alpha = F.softmax(alpha_logits, dim=-1)  # (..., n_basal)
        motive = alpha @ self.basis  # (..., d)
        return F.normalize(motive, dim=-1)

    # ------------------------------------------------------------
    # Вычисление размерности симплекса (число активных вершин - 1)
    # ------------------------------------------------------------
    def get_simplex_dim(self, alpha_logits: torch.Tensor, eps: float = 0.05) -> int:
        """Размерность симплекса по числу значимых барицентрических координат."""
        probs = F.softmax(alpha_logits, dim=-1)
        return (probs > eps).sum().item() - 1

    # ------------------------------------------------------------
    # Носитель (активные вершины) для гейта
    # ------------------------------------------------------------
    def get_active_vertices(self, alpha_logits: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
        """Бинарная маска активных вершин (носитель)."""
        probs = F.softmax(alpha_logits, dim=-1)
        return (probs > eps).float()

    # ------------------------------------------------------------
    # Добавление фрейма в историю
    # ------------------------------------------------------------
    def push_frame(self, frame: torch.Tensor, salience: float = 1.0):
        idx = self.history_ptr.item() % self.cfg.time_window
        self.history[idx] = frame
        self.frame_salience[idx] = salience
        self.history_ptr += 1

    # ------------------------------------------------------------
    # Извлечение контекста из истории
    # ------------------------------------------------------------
    def get_temporal_context(self) -> Tuple[torch.Tensor, torch.Tensor]:
        T = min(self.history_ptr.item(), self.cfg.time_window)
        if T == 0:
            return torch.zeros(self.cfg.motive_dim), torch.zeros(self.cfg.color_dim)

        hist = self.history[:T]            # (T, d)
        sal = self.frame_salience[:T]      # (T,)

        # Временные веса: чем свежее, тем больше
        t = torch.arange(T, dtype=torch.float32, device=hist.device)
        t = t - (T - 1)  # от -(T-1) до 0
        # Масштаб адаптивный: медианный масштаб времени
        med_scale = self.time_scales[self.active_mask].median()
        if med_scale == 0 or torch.isnan(med_scale):
            med_scale = torch.tensor(1.0, device=hist.device)
        # Веса: экспонента от -|t|/scale + log(salience)
        logits = -torch.abs(t) / (med_scale + 1e-8) + torch.log(sal + 1e-8)
        weights = torch.softmax(logits, dim=0)  # (T,)
        ctx = weights @ hist                     # (d,)

        # Входной спектр — взвешенное среднее первых color_dim компонент
        input_spectrum = weights @ hist[:, :self.cfg.color_dim]
        return ctx, input_spectrum

    # ------------------------------------------------------------
    # Резонансная метрика T (адаптивная)
    # ------------------------------------------------------------
    def compute_T(self, motives: torch.Tensor, spectra: torch.Tensor,
                  input_spectrum: torch.Tensor) -> torch.Tensor:
        # Геодезическое расстояние на сфере
        cos = motives @ motives.T
        dist_sq = 2.0 - 2.0 * cos

        # Масштабный коэффициент alpha адаптируется через удовлетворённость (поток Риччи)
        alpha = 1.0 / (self.dist_ma + 1e-8)  # обратное среднее расстояние
        T_base = torch.exp(-alpha * dist_sq)

        # Спектральное различие (бета — адаптивное)
        spec_diff = torch.sum((spectra.unsqueeze(1) - spectra.unsqueeze(0))**2, dim=2)
        beta = 1.0 / (self.spec_disp_ma + 1e-8)
        T_color = T_base * (1.0 + beta * spec_diff)

        # Резонанс со входным спектром
        resonance = torch.exp(-torch.sum((spectra - input_spectrum.unsqueeze(0))**2, dim=1))
        T_res = torch.outer(resonance, resonance)
        T = T_color * (1.0 + T_res)

        # Разреженность: оставляем top_k = min(2*dim, N)
        k = min(2 * self.num_basal, motives.shape[0])
        if k > 0:
            top_vals, top_idx = torch.topk(cos, k, dim=1)
            mask = torch.zeros_like(T)
            mask.scatter_(1, top_idx, 1.0)
            T = T * mask
        return T

    # ------------------------------------------------------------
    # Хирургия (гомотопическое выталкивание)
    # ------------------------------------------------------------
    def surgery(self):
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) < 2:
            return

        # Вычисляем близость по носителям
        alpha = F.softmax(self.alpha_logits[active_idx], dim=-1)
        # Носители (бинарные, eps адаптивный как доля от максимума)
        eps = 0.1 * alpha.max(dim=-1).values.mean()
        carriers = (alpha > eps).float()  # (N, n_basal)

        # Пересечение носителей
        overlap = carriers @ carriers.T  # (N, N)
        max_active = torch.max(carriers.sum(dim=1).unsqueeze(1),
                               carriers.sum(dim=1).unsqueeze(0))
        tau_matrix = 1.0 - overlap / (max_active + 1e-8)  # коразмерность

        # Также учитываем близость спектров
        spec_dist = torch.sum((self.spectrum[active_idx].unsqueeze(1) -
                               self.spectrum[active_idx].unsqueeze(0))**2, dim=2)
        # Комбинированная мера для склейки (чем меньше, тем ближе)
        merge_score = tau_matrix + spec_dist * 0.1

        # Исключаем самопары
        merge_score = merge_score + torch.eye(len(active_idx), device=merge_score.device) * 1e9

        # Порог: медиана по всем парам
        thresh = merge_score.median()
        min_val, min_idx = merge_score.min(dim=1)
        val, row = min_val.min(dim=0)
        col = min_idx[row]
        if val > thresh:
            return  # нет пар достаточно близких

        i = active_idx[row].item()
        j = active_idx[col].item()

        # Склейка: усреднение альфа-логитов и спектров
        new_alpha_logits = (self.alpha_logits[i] + self.alpha_logits[j]) / 2.0
        new_spectrum = (self.spectrum[i] + self.spectrum[j]) / 2.0

        self.alpha_logits.data[i] = new_alpha_logits
        self.spectrum.data[i] = new_spectrum
        self.active_mask[j] = False
        self.age[i] = 0  # обнуляем возраст

    # ------------------------------------------------------------
    # Устаревание: сдвиг спектра от центра
    # ------------------------------------------------------------
    def apply_aging(self):
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) == 0:
            return
        # Центр спектров
        center = self.spectrum[active_idx].mean(dim=0)
        # Сдвиг для всех активных слотов
        lr_age = 1.0 / (len(active_idx) + 1.0)
        for idx in active_idx:
            if self.age[idx] > 0:
                self.spectrum.data[idx] += lr_age * (self.spectrum.data[idx] - center)
                self.spectrum.data[idx].clamp_(0.0, 1.0)
                self.age[idx] += 1

    # ------------------------------------------------------------
    # Синестезия: расширение базиса при необходимости
    # ------------------------------------------------------------
    def expand_basis(self, frame: torch.Tensor):
        """Если остаток проекции велик, добавляем новую вершину."""
        # Проекция на текущий базис
        with torch.no_grad():
            proj = frame @ self.basis.T  # (d,) @ (n_basal, d)^T -> (n_basal,)
            reconstruction = proj @ self.basis
            residual = frame - reconstruction
            res_norm = torch.norm(residual)

            # Адаптивный порог: скользящее среднее + 2*MAD
            self.residual_ma = 0.9 * self.residual_ma + 0.1 * res_norm
            threshold = self.residual_ma * 2.0

            if res_norm > threshold and self.num_basal < self.cfg.max_slots:
                # Добавляем новую ортонормированную вершину
                new_vec = F.normalize(residual, dim=0)
                # Расширяем базис
                self.basis = nn.Parameter(torch.cat([self.basis.data, new_vec.unsqueeze(0)], dim=0))
                # Расширяем alpha_logits (добавляем нулевой столбец)
                self.alpha_logits = nn.Parameter(
                    torch.cat([self.alpha_logits.data,
                               torch.zeros(self.cfg.max_slots, 1, device=self.alpha_logits.device)], dim=1)
                )
                self.num_basal += 1
                # Обновляем time_scales (добавляем среднее значение)
                self.time_scales = nn.Parameter(
                    torch.cat([self.time_scales.data,
                               torch.ones(1, device=self.time_scales.device) * self.time_scales.mean()])
                )
                self.time_scales.requires_grad = False

    # ------------------------------------------------------------
    # Основной шаг
    # ------------------------------------------------------------
    def forward(self, frame: torch.Tensor, dream: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if dream:
            # Во сне вход — усреднённый базальный мотив
            persona = self.basis.mean(dim=0)
            frame = torch.zeros_like(persona)

        # Синестезия: проверка на новое измерение (только не во сне)
        if not dream:
            self.expand_basis(frame)

        # История
        # Важность фрейма = размерность спроецированного симплекса
        proj_logits = frame @ self.basis.T  # (n_basal,)
        salience = max(1, (F.softmax(proj_logits, dim=-1) > 0.05).sum().item())
        self.push_frame(frame, salience=float(salience))
        ctx, input_spectrum = self.get_temporal_context()

        # Нормировка базиса и мотивов
        with torch.no_grad():
            self.basis.data = F.normalize(self.basis.data, dim=1)
        motives = self.get_motive(self.alpha_logits)  # все слоты, включая неактивные

        # Метрика T (только по активным)
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        active_motives = motives[active_idx]
        active_spectra = self.spectrum[active_idx]
        T = self.compute_T(active_motives, active_spectra, input_spectrum)

        # Резонанс и температура
        resonance = torch.exp(-torch.sum((active_spectra - input_spectrum.unsqueeze(0))**2, dim=1))
        temperature = self.cfg.initial_temperature / (1.0 + resonance.mean())  # адаптив

        # Внимание: контекст к активным слотам
        logits = (ctx @ active_motives.T) / temperature
        # Добавляем бонус за низкую размерность (базальные более стабильны)
        dims = torch.tensor([self.get_simplex_dim(self.alpha_logits[idx]) for idx in active_idx],
                            device=logits.device).float()
        basal_bonus = torch.exp(-dims) * 0.1
        attn = torch.softmax(logits + basal_bonus, dim=-1)

        # Память: взвешенная сумма мотивов
        mem_contrib = attn @ active_motives

        # Носители для гейта
        # Контекст: проекция ctx на базис
        ctx_proj = ctx @ self.basis.T  # (n_basal,)
        active_ctx = self.get_active_vertices(ctx_proj)
        # Память: средний носитель активных слотов с учётом внимания
        active_carriers = torch.stack([self.get_active_vertices(self.alpha_logits[idx])
                                       for idx in active_idx])
        active_mem = (attn @ active_carriers.float()).clamp(0, 1)  # мягкое среднее
        active_mem = (active_mem > 0.5).float()  # бинаризуем

        C0 = self.gate(ctx, mem_contrib, active_ctx, active_mem)

        # Удовлетворённость
        S_true = F.cosine_similarity(C0, self.target, dim=0)
        self.sat_ma = 0.9 * self.sat_ma + 0.1 * S_true.detach()

        # Поток Риччи (обновление адаптивных статистик)
        with torch.no_grad():
            # Обновляем среднее расстояние мотивов
            if len(active_idx) > 1:
                dists = 2.0 - 2.0 * (active_motives @ active_motives.T)
                self.dist_ma = 0.9 * self.dist_ma + 0.1 * dists.mean()
            # Обновляем дисперсию спектров
            if len(active_idx) > 1:
                spec_var = torch.var(active_spectra, dim=0).mean()
                self.spec_disp_ma = 0.9 * self.spec_disp_ma + 0.1 * spec_var

        # Градиенты энергии Риччи (упрощённо: минимизируем T * dist)
        energy = (T * (2.0 - 2.0 * (active_motives @ active_motives.T))).sum()
        grad_alpha = torch.autograd.grad(energy, self.alpha_logits, retain_graph=True, create_graph=False)[0]
        grad_spec = torch.autograd.grad(energy, self.spectrum, retain_graph=True, create_graph=False)[0]

        with torch.no_grad():
            # Обновление только активных
            lr_alpha = 0.01 / (len(active_idx) + 1.0)
            lr_spec = 0.01 / (len(active_idx) + 1.0)
            self.alpha_logits[active_idx] -= lr_alpha * grad_alpha[active_idx]
            self.spectrum[active_idx] -= lr_spec * grad_spec[active_idx]
            self.spectrum.clamp_(0.0, 1.0)

        # Хирургия (один шаг) и устаревание
        self.surgery()
        self.apply_aging()

        # Обновление временных масштабов: обратно пропорционально размерности
        with torch.no_grad():
            for idx in active_idx:
                dim = self.get_simplex_dim(self.alpha_logits[idx])
                self.time_scales[idx] = torch.exp(torch.tensor(-float(dim)))

        # Аффективный выход
        affective_out = self.action(C0, self.target)
        global_affect = torch.norm(affective_out[:self.num_basal])  # динамически
        return C0, S_true, affective_out, global_affect

    # ------------------------------------------------------------
    # Микросон: многократная хирургия + стабилизация
    # ------------------------------------------------------------
    def microsleep(self, steps: int = 3):
        for _ in range(steps):
            self.forward(torch.zeros(self.cfg.motive_dim), dream=True)
        # Зарядка: обнуляем возраст и сдвиг спектров
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) > 0:
            center = self.spectrum[active_idx].mean(dim=0)
            with torch.no_grad():
                for idx in active_idx:
                    self.spectrum[idx] = center  # возврат к центру
                    self.age[idx] = 0
        # Проверка связности: если есть изолированные вершины базиса, создаём мостик
        # (упрощённо: убеждаемся, что все вершины используются хотя бы одним слотом)
        alpha = F.softmax(self.alpha_logits[active_idx], dim=-1)
        used = (alpha > 0.05).any(dim=0)  # (n_basal,)
        if not used.all():
            # Добавляем слот, связывающий неиспользуемые вершины с центром
            for i in range(self.num_basal):
                if not used[i]:
                    # Находим неактивный слот и делаем его мостиком
                    inactive = (~self.active_mask).nonzero(as_tuple=True)[0]
                    if len(inactive) > 0:
                        idx = inactive[0].item()
                        self.alpha_logits.data[idx] = torch.zeros(self.num_basal)
                        self.alpha_logits.data[idx, i] = 1.0
                        self.alpha_logits.data[idx, used.nonzero()[0]] = 1.0  # связь с центром
                        self.spectrum.data[idx] = center
                        self.active_mask[idx] = True
