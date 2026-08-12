import numpy as np


class ConvexBody:
    def support(self, e):
        raise NotImplementedError

    def argmax(self, e):
        raise NotImplementedError


class Disc(ConvexBody):
    def support(self, e):
        return np.linalg.norm(e, axis=-1)

    def argmax(self, e):
        n = np.linalg.norm(e, axis=-1, keepdims=True)
        return np.divide(e, n, out=np.zeros_like(e), where=n > 0)


class Polygon(ConvexBody):
    def __init__(self, verts):
        self.verts = np.asarray(verts, dtype=float)

    @classmethod
    def regular_inscribed(cls, n, theta=0.0):
        if n % 2 != 0:
            raise ValueError(f"n must be even for central symmetry, got {n}")
        ang = theta + 2.0 * np.pi * np.arange(n) / n
        return cls(np.stack([np.cos(ang), np.sin(ang)], axis=-1))

    @classmethod
    def regular_circumscribed(cls, n, theta=0.0):
        if n % 2 != 0:
            raise ValueError(f"n must be even for central symmetry, got {n}")
        r = 1.0 / np.cos(np.pi / n)
        ang = theta + np.pi / n + 2.0 * np.pi * np.arange(n) / n
        return cls(r * np.stack([np.cos(ang), np.sin(ang)], axis=-1))

    @classmethod
    def rectangle(cls, a, b, theta=0.0):
        v = np.array([[a, b], [-a, b], [-a, -b], [a, -b]], dtype=float)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        return cls(v @ rot.T)

    @classmethod
    def octagon_circumscribed(cls, theta=0.0):
        return cls.regular_circumscribed(8, theta)

    def support(self, e):
        e = np.asarray(e, dtype=float)
        return np.max(e @ self.verts.T, axis=-1)

    def argmax(self, e):
        e = np.asarray(e, dtype=float)
        k = np.argmax(e @ self.verts.T, axis=-1)
        return self.verts[k]


def gramian(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.array([[x @ x, x @ y], [x @ y, y @ y]])


def to_pq(x, y):
    """p = ||x||^2/||y||^4, q = <x,y>/||y||^3."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ny = np.linalg.norm(y)
    if ny <= 0.0:
        raise ValueError("y must be nonzero to define (p, q)")
    return float((x @ x) / ny**4), float((x @ y) / ny**3)


def from_pq(p, q, ny=1.0):
    if p * 1.0 < q * q:
        raise ValueError(f"infeasible invariants: p={p} < q^2={q * q}")
    y = np.array([ny, 0.0])
    x1 = q * ny
    x2 = np.sqrt(max(p * ny**2 - q * q * ny**2, 0.0))
    return np.array([x1, x2]), y
