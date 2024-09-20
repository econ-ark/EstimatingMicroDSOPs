!(function (t, e) {
  "object" == typeof exports && "undefined" != typeof module
    ? (module.exports = e())
    : "function" == typeof define && define.amd
      ? define(e)
      : ((t =
          "undefined" != typeof globalThis
            ? globalThis
            : t || self).RevealNotes = e());
})(this, function () {
  "use strict";
  var t =
      "undefined" != typeof globalThis
        ? globalThis
        : "undefined" != typeof window
          ? window
          : "undefined" != typeof global
            ? global
            : "undefined" != typeof self
              ? self
              : {},
    e = function (t) {
      return t && t.Math == Math && t;
    },
    n =
      e("object" == typeof globalThis && globalThis) ||
      e("object" == typeof window && window) ||
      e("object" == typeof self && self) ||
      e("object" == typeof t && t) ||
      (function () {
        return this;
      })() ||
      Function("return this")(),
    r = {},
    i = function (t) {
      try {
        return !!t();
      } catch (t) {
        return !0;
      }
    },
    u = !i(function () {
      return (
        7 !=
        Object.defineProperty({}, 1, {
          get: function () {
            return 7;
          },
        })[1]
      );
    }),
    a = {},
    o = {}.propertyIsEnumerable,
    s = Object.getOwnPropertyDescriptor,
    l = s && !o.call({ 1: 2 }, 1);
  a.f = l
    ? function (t) {
        var e = s(this, t);
        return !!e && e.enumerable;
      }
    : o;
  var c = function (t, e) {
      return {
        enumerable: !(1 & t),
        configurable: !(2 & t),
        writable: !(4 & t),
        value: e,
      };
    },
    p = {}.toString,
    d = function (t) {
      return p.call(t).slice(8, -1);
    },
    f = d,
    h = "".split,
    g = i(function () {
      return !Object("z").propertyIsEnumerable(0);
    })
      ? function (t) {
          return "String" == f(t) ? h.call(t, "") : Object(t);
        }
      : Object,
    D = function (t) {
      if (null == t) throw TypeError("Can't call method on " + t);
      return t;
    },
    m = g,
    v = D,
    y = function (t) {
      return m(v(t));
    },
    k = function (t) {
      return "object" == typeof t ? null !== t : "function" == typeof t;
    },
    E = k,
    x = function (t, e) {
      if (!E(t)) return t;
      var n, r;
      if (e && "function" == typeof (n = t.toString) && !E((r = n.call(t))))
        return r;
      if ("function" == typeof (n = t.valueOf) && !E((r = n.call(t)))) return r;
      if (!e && "function" == typeof (n = t.toString) && !E((r = n.call(t))))
        return r;
      throw TypeError("Can't convert object to primitive value");
    },
    A = D,
    b = function (t) {
      return Object(A(t));
    },
    w = b,
    C = {}.hasOwnProperty,
    F = function (t, e) {
      return C.call(w(t), e);
    },
    S = k,
    B = n.document,
    T = S(B) && S(B.createElement),
    _ = function (t) {
      return T ? B.createElement(t) : {};
    },
    z = _,
    R =
      !u &&
      !i(function () {
        return (
          7 !=
          Object.defineProperty(z("div"), "a", {
            get: function () {
              return 7;
            },
          }).a
        );
      }),
    I = u,
    L = a,
    O = c,
    $ = y,
    P = x,
    M = F,
    j = R,
    N = Object.getOwnPropertyDescriptor;
  r.f = I
    ? N
    : function (t, e) {
        if (((t = $(t)), (e = P(e, !0)), j))
          try {
            return N(t, e);
          } catch (t) {}
        if (M(t, e)) return O(!L.f.call(t, e), t[e]);
      };
  var U = {},
    q = k,
    Z = function (t) {
      if (!q(t)) throw TypeError(String(t) + " is not an object");
      return t;
    },
    H = u,
    W = R,
    J = Z,
    V = x,
    K = Object.defineProperty;
  U.f = H
    ? K
    : function (t, e, n) {
        if ((J(t), (e = V(e, !0)), J(n), W))
          try {
            return K(t, e, n);
          } catch (t) {}
        if ("get" in n || "set" in n)
          throw TypeError("Accessors not supported");
        return "value" in n && (t[e] = n.value), t;
      };
  var Q = U,
    G = c,
    Y = u
      ? function (t, e, n) {
          return Q.f(t, e, G(1, n));
        }
      : function (t, e, n) {
          return (t[e] = n), t;
        },
    X = { exports: {} },
    tt = n,
    et = Y,
    nt = function (t, e) {
      try {
        et(tt, t, e);
      } catch (n) {
        tt[t] = e;
      }
      return e;
    },
    rt = nt,
    it = "__core-js_shared__",
    ut = n[it] || rt(it, {}),
    at = ut,
    ot = Function.toString;
  "function" != typeof at.inspectSource &&
    (at.inspectSource = function (t) {
      return ot.call(t);
    });
  var st = at.inspectSource,
    lt = st,
    ct = n.WeakMap,
    pt = "function" == typeof ct && /native code/.test(lt(ct)),
    dt = { exports: {} },
    ft = ut;
  (dt.exports = function (t, e) {
    return ft[t] || (ft[t] = void 0 !== e ? e : {});
  })("versions", []).push({
    version: "3.12.1",
    mode: "global",
    copyright: "© 2021 Denis Pushkarev (zloirock.ru)",
  });
  var ht,
    gt,
    Dt,
    mt = 0,
    vt = Math.random(),
    yt = function (t) {
      return (
        "Symbol(" +
        String(void 0 === t ? "" : t) +
        ")_" +
        (++mt + vt).toString(36)
      );
    },
    kt = dt.exports,
    Et = yt,
    xt = kt("keys"),
    At = function (t) {
      return xt[t] || (xt[t] = Et(t));
    },
    bt = {},
    wt = pt,
    Ct = k,
    Ft = Y,
    St = F,
    Bt = ut,
    Tt = At,
    _t = bt,
    zt = "Object already initialized",
    Rt = n.WeakMap;
  if (wt || Bt.state) {
    var It = Bt.state || (Bt.state = new Rt()),
      Lt = It.get,
      Ot = It.has,
      $t = It.set;
    (ht = function (t, e) {
      if (Ot.call(It, t)) throw new TypeError(zt);
      return (e.facade = t), $t.call(It, t, e), e;
    }),
      (gt = function (t) {
        return Lt.call(It, t) || {};
      }),
      (Dt = function (t) {
        return Ot.call(It, t);
      });
  } else {
    var Pt = Tt("state");
    (_t[Pt] = !0),
      (ht = function (t, e) {
        if (St(t, Pt)) throw new TypeError(zt);
        return (e.facade = t), Ft(t, Pt, e), e;
      }),
      (gt = function (t) {
        return St(t, Pt) ? t[Pt] : {};
      }),
      (Dt = function (t) {
        return St(t, Pt);
      });
  }
  var Mt = {
      set: ht,
      get: gt,
      has: Dt,
      enforce: function (t) {
        return Dt(t) ? gt(t) : ht(t, {});
      },
      getterFor: function (t) {
        return function (e) {
          var n;
          if (!Ct(e) || (n = gt(e)).type !== t)
            throw TypeError("Incompatible receiver, " + t + " required");
          return n;
        };
      },
    },
    jt = n,
    Nt = Y,
    Ut = F,
    qt = nt,
    Zt = st,
    Ht = Mt.get,
    Wt = Mt.enforce,
    Jt = String(String).split("String");
  (X.exports = function (t, e, n, r) {
    var i,
      u = !!r && !!r.unsafe,
      a = !!r && !!r.enumerable,
      o = !!r && !!r.noTargetGet;
    "function" == typeof n &&
      ("string" != typeof e || Ut(n, "name") || Nt(n, "name", e),
      (i = Wt(n)).source ||
        (i.source = Jt.join("string" == typeof e ? e : ""))),
      t !== jt
        ? (u ? !o && t[e] && (a = !0) : delete t[e],
          a ? (t[e] = n) : Nt(t, e, n))
        : a
          ? (t[e] = n)
          : qt(e, n);
  })(Function.prototype, "toString", function () {
    return ("function" == typeof this && Ht(this).source) || Zt(this);
  });
  var Vt = n,
    Kt = n,
    Qt = function (t) {
      return "function" == typeof t ? t : void 0;
    },
    Gt = function (t, e) {
      return arguments.length < 2
        ? Qt(Vt[t]) || Qt(Kt[t])
        : (Vt[t] && Vt[t][e]) || (Kt[t] && Kt[t][e]);
    },
    Yt = {},
    Xt = Math.ceil,
    te = Math.floor,
    ee = function (t) {
      return isNaN((t = +t)) ? 0 : (t > 0 ? te : Xt)(t);
    },
    ne = ee,
    re = Math.min,
    ie = function (t) {
      return t > 0 ? re(ne(t), 9007199254740991) : 0;
    },
    ue = ee,
    ae = Math.max,
    oe = Math.min,
    se = function (t, e) {
      var n = ue(t);
      return n < 0 ? ae(n + e, 0) : oe(n, e);
    },
    le = y,
    ce = ie,
    pe = se,
    de = function (t) {
      return function (e, n, r) {
        var i,
          u = le(e),
          a = ce(u.length),
          o = pe(r, a);
        if (t && n != n) {
          for (; a > o; ) if ((i = u[o++]) != i) return !0;
        } else
          for (; a > o; o++)
            if ((t || o in u) && u[o] === n) return t || o || 0;
        return !t && -1;
      };
    },
    fe = { includes: de(!0), indexOf: de(!1) },
    he = F,
    ge = y,
    De = fe.indexOf,
    me = bt,
    ve = function (t, e) {
      var n,
        r = ge(t),
        i = 0,
        u = [];
      for (n in r) !he(me, n) && he(r, n) && u.push(n);
      for (; e.length > i; ) he(r, (n = e[i++])) && (~De(u, n) || u.push(n));
      return u;
    },
    ye = [
      "constructor",
      "hasOwnProperty",
      "isPrototypeOf",
      "propertyIsEnumerable",
      "toLocaleString",
      "toString",
      "valueOf",
    ],
    ke = ve,
    Ee = ye.concat("length", "prototype");
  Yt.f =
    Object.getOwnPropertyNames ||
    function (t) {
      return ke(t, Ee);
    };
  var xe = {};
  xe.f = Object.getOwnPropertySymbols;
  var Ae = Yt,
    be = xe,
    we = Z,
    Ce =
      Gt("Reflect", "ownKeys") ||
      function (t) {
        var e = Ae.f(we(t)),
          n = be.f;
        return n ? e.concat(n(t)) : e;
      },
    Fe = F,
    Se = Ce,
    Be = r,
    Te = U,
    _e = i,
    ze = /#|\.prototype\./,
    Re = function (t, e) {
      var n = Le[Ie(t)];
      return n == $e || (n != Oe && ("function" == typeof e ? _e(e) : !!e));
    },
    Ie = (Re.normalize = function (t) {
      return String(t).replace(ze, ".").toLowerCase();
    }),
    Le = (Re.data = {}),
    Oe = (Re.NATIVE = "N"),
    $e = (Re.POLYFILL = "P"),
    Pe = Re,
    Me = n,
    je = r.f,
    Ne = Y,
    Ue = X.exports,
    qe = nt,
    Ze = function (t, e) {
      for (var n = Se(e), r = Te.f, i = Be.f, u = 0; u < n.length; u++) {
        var a = n[u];
        Fe(t, a) || r(t, a, i(e, a));
      }
    },
    He = Pe,
    We = function (t, e) {
      var n,
        r,
        i,
        u,
        a,
        o = t.target,
        s = t.global,
        l = t.stat;
      if ((n = s ? Me : l ? Me[o] || qe(o, {}) : (Me[o] || {}).prototype))
        for (r in e) {
          if (
            ((u = e[r]),
            (i = t.noTargetGet ? (a = je(n, r)) && a.value : n[r]),
            !He(s ? r : o + (l ? "." : "#") + r, t.forced) && void 0 !== i)
          ) {
            if (typeof u == typeof i) continue;
            Ze(u, i);
          }
          (t.sham || (i && i.sham)) && Ne(u, "sham", !0), Ue(n, r, u, t);
        }
    },
    Je = Z,
    Ve = function () {
      var t = Je(this),
        e = "";
      return (
        t.global && (e += "g"),
        t.ignoreCase && (e += "i"),
        t.multiline && (e += "m"),
        t.dotAll && (e += "s"),
        t.unicode && (e += "u"),
        t.sticky && (e += "y"),
        e
      );
    },
    Ke = {},
    Qe = i;
  function Ge(t, e) {
    return RegExp(t, e);
  }
  (Ke.UNSUPPORTED_Y = Qe(function () {
    var t = Ge("a", "y");
    return (t.lastIndex = 2), null != t.exec("abcd");
  })),
    (Ke.BROKEN_CARET = Qe(function () {
      var t = Ge("^r", "gy");
      return (t.lastIndex = 2), null != t.exec("str");
    }));
  var Ye = Ve,
    Xe = Ke,
    tn = dt.exports,
    en = RegExp.prototype.exec,
    nn = tn("native-string-replace", String.prototype.replace),
    rn = en,
    un = (function () {
      var t = /a/,
        e = /b*/g;
      return (
        en.call(t, "a"), en.call(e, "a"), 0 !== t.lastIndex || 0 !== e.lastIndex
      );
    })(),
    an = Xe.UNSUPPORTED_Y || Xe.BROKEN_CARET,
    on = void 0 !== /()??/.exec("")[1];
  (un || on || an) &&
    (rn = function (t) {
      var e,
        n,
        r,
        i,
        u = this,
        a = an && u.sticky,
        o = Ye.call(u),
        s = u.source,
        l = 0,
        c = t;
      return (
        a &&
          (-1 === (o = o.replace("y", "")).indexOf("g") && (o += "g"),
          (c = String(t).slice(u.lastIndex)),
          u.lastIndex > 0 &&
            (!u.multiline || (u.multiline && "\n" !== t[u.lastIndex - 1])) &&
            ((s = "(?: " + s + ")"), (c = " " + c), l++),
          (n = new RegExp("^(?:" + s + ")", o))),
        on && (n = new RegExp("^" + s + "$(?!\\s)", o)),
        un && (e = u.lastIndex),
        (r = en.call(a ? n : u, c)),
        a
          ? r
            ? ((r.input = r.input.slice(l)),
              (r[0] = r[0].slice(l)),
              (r.index = u.lastIndex),
              (u.lastIndex += r[0].length))
            : (u.lastIndex = 0)
          : un && r && (u.lastIndex = u.global ? r.index + r[0].length : e),
        on &&
          r &&
          r.length > 1 &&
          nn.call(r[0], n, function () {
            for (i = 1; i < arguments.length - 2; i++)
              void 0 === arguments[i] && (r[i] = void 0);
          }),
        r
      );
    });
  var sn = rn;
  We({ target: "RegExp", proto: !0, forced: /./.exec !== sn }, { exec: sn });
  var ln,
    cn,
    pn = Gt("navigator", "userAgent") || "",
    dn = n.process,
    fn = dn && dn.versions,
    hn = fn && fn.v8;
  hn
    ? (cn = (ln = hn.split("."))[0] < 4 ? 1 : ln[0] + ln[1])
    : pn &&
      (!(ln = pn.match(/Edge\/(\d+)/)) || ln[1] >= 74) &&
      (ln = pn.match(/Chrome\/(\d+)/)) &&
      (cn = ln[1]);
  var gn = cn && +cn,
    Dn = gn,
    mn = i,
    vn =
      !!Object.getOwnPropertySymbols &&
      !mn(function () {
        return !String(Symbol()) || (!Symbol.sham && Dn && Dn < 41);
      }),
    yn = vn && !Symbol.sham && "symbol" == typeof Symbol.iterator,
    kn = n,
    En = dt.exports,
    xn = F,
    An = yt,
    bn = vn,
    wn = yn,
    Cn = En("wks"),
    Fn = kn.Symbol,
    Sn = wn ? Fn : (Fn && Fn.withoutSetter) || An,
    Bn = function (t) {
      return (
        (xn(Cn, t) && (bn || "string" == typeof Cn[t])) ||
          (bn && xn(Fn, t) ? (Cn[t] = Fn[t]) : (Cn[t] = Sn("Symbol." + t))),
        Cn[t]
      );
    },
    Tn = X.exports,
    _n = sn,
    zn = i,
    Rn = Bn,
    In = Y,
    Ln = Rn("species"),
    On = RegExp.prototype,
    $n = !zn(function () {
      var t = /./;
      return (
        (t.exec = function () {
          var t = [];
          return (t.groups = { a: "7" }), t;
        }),
        "7" !== "".replace(t, "$<a>")
      );
    }),
    Pn = "$0" === "a".replace(/./, "$0"),
    Mn = Rn("replace"),
    jn = !!/./[Mn] && "" === /./[Mn]("a", "$0"),
    Nn = !zn(function () {
      var t = /(?:)/,
        e = t.exec;
      t.exec = function () {
        return e.apply(this, arguments);
      };
      var n = "ab".split(t);
      return 2 !== n.length || "a" !== n[0] || "b" !== n[1];
    }),
    Un = function (t, e, n, r) {
      var i = Rn(t),
        u = !zn(function () {
          var e = {};
          return (
            (e[i] = function () {
              return 7;
            }),
            7 != ""[t](e)
          );
        }),
        a =
          u &&
          !zn(function () {
            var e = !1,
              n = /a/;
            return (
              "split" === t &&
                (((n = {}).constructor = {}),
                (n.constructor[Ln] = function () {
                  return n;
                }),
                (n.flags = ""),
                (n[i] = /./[i])),
              (n.exec = function () {
                return (e = !0), null;
              }),
              n[i](""),
              !e
            );
          });
      if (
        !u ||
        !a ||
        ("replace" === t && (!$n || !Pn || jn)) ||
        ("split" === t && !Nn)
      ) {
        var o = /./[i],
          s = n(
            i,
            ""[t],
            function (t, e, n, r, i) {
              var a = e.exec;
              return a === _n || a === On.exec
                ? u && !i
                  ? { done: !0, value: o.call(e, n, r) }
                  : { done: !0, value: t.call(n, e, r) }
                : { done: !1 };
            },
            {
              REPLACE_KEEPS_$0: Pn,
              REGEXP_REPLACE_SUBSTITUTES_UNDEFINED_CAPTURE: jn,
            },
          ),
          l = s[0],
          c = s[1];
        Tn(String.prototype, t, l),
          Tn(
            On,
            i,
            2 == e
              ? function (t, e) {
                  return c.call(t, this, e);
                }
              : function (t) {
                  return c.call(t, this);
                },
          );
      }
      r && In(On[i], "sham", !0);
    },
    qn =
      Object.is ||
      function (t, e) {
        return t === e ? 0 !== t || 1 / t == 1 / e : t != t && e != e;
      },
    Zn = d,
    Hn = sn,
    Wn = function (t, e) {
      var n = t.exec;
      if ("function" == typeof n) {
        var r = n.call(t, e);
        if ("object" != typeof r)
          throw TypeError(
            "RegExp exec method returned something other than an Object or null",
          );
        return r;
      }
      if ("RegExp" !== Zn(t))
        throw TypeError("RegExp#exec called on incompatible receiver");
      return Hn.call(t, e);
    },
    Jn = Z,
    Vn = D,
    Kn = qn,
    Qn = Wn;
  Un("search", 1, function (t, e, n) {
    return [
      function (e) {
        var n = Vn(this),
          r = null == e ? void 0 : e[t];
        return void 0 !== r ? r.call(e, n) : new RegExp(e)[t](String(n));
      },
      function (t) {
        var r = n(e, t, this);
        if (r.done) return r.value;
        var i = Jn(t),
          u = String(this),
          a = i.lastIndex;
        Kn(a, 0) || (i.lastIndex = 0);
        var o = Qn(i, u);
        return (
          Kn(i.lastIndex, a) || (i.lastIndex = a), null === o ? -1 : o.index
        );
      },
    ];
  });
  var Gn = ee,
    Yn = D,
    Xn = function (t) {
      return function (e, n) {
        var r,
          i,
          u = String(Yn(e)),
          a = Gn(n),
          o = u.length;
        return a < 0 || a >= o
          ? t
            ? ""
            : void 0
          : (r = u.charCodeAt(a)) < 55296 ||
              r > 56319 ||
              a + 1 === o ||
              (i = u.charCodeAt(a + 1)) < 56320 ||
              i > 57343
            ? t
              ? u.charAt(a)
              : r
            : t
              ? u.slice(a, a + 2)
              : i - 56320 + ((r - 55296) << 10) + 65536;
      };
    },
    tr = { codeAt: Xn(!1), charAt: Xn(!0) }.charAt,
    er = function (t, e, n) {
      return e + (n ? tr(t, e).length : 1);
    },
    nr = Z,
    rr = ie,
    ir = D,
    ur = er,
    ar = Wn;
  function or(t, e) {
    if (!(t instanceof e))
      throw new TypeError("Cannot call a class as a function");
  }
  function sr(t, e) {
    for (var n = 0; n < e.length; n++) {
      var r = e[n];
      (r.enumerable = r.enumerable || !1),
        (r.configurable = !0),
        "value" in r && (r.writable = !0),
        Object.defineProperty(t, r.key, r);
    }
  }
  function lr(t, e, n) {
    return e && sr(t.prototype, e), n && sr(t, n), t;
  }
  function cr(t, e) {
    return (
      (function (t) {
        if (Array.isArray(t)) return t;
      })(t) ||
      (function (t, e) {
        var n =
          t &&
          (("undefined" != typeof Symbol && t[Symbol.iterator]) ||
            t["@@iterator"]);
        if (null == n) return;
        var r,
          i,
          u = [],
          a = !0,
          o = !1;
        try {
          for (
            n = n.call(t);
            !(a = (r = n.next()).done) &&
            (u.push(r.value), !e || u.length !== e);
            a = !0
          );
        } catch (t) {
          (o = !0), (i = t);
        } finally {
          try {
            a || null == n.return || n.return();
          } finally {
            if (o) throw i;
          }
        }
        return u;
      })(t, e) ||
      pr(t, e) ||
      (function () {
        throw new TypeError(
          "Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.",
        );
      })()
    );
  }
  function pr(t, e) {
    if (t) {
      if ("string" == typeof t) return dr(t, e);
      var n = Object.prototype.toString.call(t).slice(8, -1);
      return (
        "Object" === n && t.constructor && (n = t.constructor.name),
        "Map" === n || "Set" === n
          ? Array.from(t)
          : "Arguments" === n ||
              /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)
            ? dr(t, e)
            : void 0
      );
    }
  }
  function dr(t, e) {
    (null == e || e > t.length) && (e = t.length);
    for (var n = 0, r = new Array(e); n < e; n++) r[n] = t[n];
    return r;
  }
  function fr(t, e) {
    var n =
      ("undefined" != typeof Symbol && t[Symbol.iterator]) || t["@@iterator"];
    if (!n) {
      if (
        Array.isArray(t) ||
        (n = pr(t)) ||
        (e && t && "number" == typeof t.length)
      ) {
        n && (t = n);
        var r = 0,
          i = function () {};
        return {
          s: i,
          n: function () {
            return r >= t.length ? { done: !0 } : { done: !1, value: t[r++] };
          },
          e: function (t) {
            throw t;
          },
          f: i,
        };
      }
      throw new TypeError(
        "Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.",
      );
    }
    var u,
      a = !0,
      o = !1;
    return {
      s: function () {
        n = n.call(t);
      },
      n: function () {
        var t = n.next();
        return (a = t.done), t;
      },
      e: function (t) {
        (o = !0), (u = t);
      },
      f: function () {
        try {
          a || null == n.return || n.return();
        } finally {
          if (o) throw u;
        }
      },
    };
  }
  Un("match", 1, function (t, e, n) {
    return [
      function (e) {
        var n = ir(this),
          r = null == e ? void 0 : e[t];
        return void 0 !== r ? r.call(e, n) : new RegExp(e)[t](String(n));
      },
      function (t) {
        var r = n(e, t, this);
        if (r.done) return r.value;
        var i = nr(t),
          u = String(this);
        if (!i.global) return ar(i, u);
        var a = i.unicode;
        i.lastIndex = 0;
        for (var o, s = [], l = 0; null !== (o = ar(i, u)); ) {
          var c = String(o[0]);
          (s[l] = c),
            "" === c && (i.lastIndex = ur(u, rr(i.lastIndex), a)),
            l++;
        }
        return 0 === l ? null : s;
      },
    ];
  });
  var hr = b,
    gr = Math.floor,
    Dr = "".replace,
    mr = /\$([$&'`]|\d{1,2}|<[^>]*>)/g,
    vr = /\$([$&'`]|\d{1,2})/g,
    yr = Un,
    kr = Z,
    Er = ie,
    xr = ee,
    Ar = D,
    br = er,
    wr = function (t, e, n, r, i, u) {
      var a = n + t.length,
        o = r.length,
        s = vr;
      return (
        void 0 !== i && ((i = hr(i)), (s = mr)),
        Dr.call(u, s, function (u, s) {
          var l;
          switch (s.charAt(0)) {
            case "$":
              return "$";
            case "&":
              return t;
            case "`":
              return e.slice(0, n);
            case "'":
              return e.slice(a);
            case "<":
              l = i[s.slice(1, -1)];
              break;
            default:
              var c = +s;
              if (0 === c) return u;
              if (c > o) {
                var p = gr(c / 10);
                return 0 === p
                  ? u
                  : p <= o
                    ? void 0 === r[p - 1]
                      ? s.charAt(1)
                      : r[p - 1] + s.charAt(1)
                    : u;
              }
              l = r[c - 1];
          }
          return void 0 === l ? "" : l;
        })
      );
    },
    Cr = Wn,
    Fr = Math.max,
    Sr = Math.min;
  yr("replace", 2, function (t, e, n, r) {
    var i = r.REGEXP_REPLACE_SUBSTITUTES_UNDEFINED_CAPTURE,
      u = r.REPLACE_KEEPS_$0,
      a = i ? "$" : "$0";
    return [
      function (n, r) {
        var i = Ar(this),
          u = null == n ? void 0 : n[t];
        return void 0 !== u ? u.call(n, i, r) : e.call(String(i), n, r);
      },
      function (t, r) {
        if ((!i && u) || ("string" == typeof r && -1 === r.indexOf(a))) {
          var o = n(e, t, this, r);
          if (o.done) return o.value;
        }
        var s = kr(t),
          l = String(this),
          c = "function" == typeof r;
        c || (r = String(r));
        var p = s.global;
        if (p) {
          var d = s.unicode;
          s.lastIndex = 0;
        }
        for (var f = []; ; ) {
          var h = Cr(s, l);
          if (null === h) break;
          if ((f.push(h), !p)) break;
          "" === String(h[0]) && (s.lastIndex = br(l, Er(s.lastIndex), d));
        }
        for (var g, D = "", m = 0, v = 0; v < f.length; v++) {
          h = f[v];
          for (
            var y = String(h[0]),
              k = Fr(Sr(xr(h.index), l.length), 0),
              E = [],
              x = 1;
            x < h.length;
            x++
          )
            E.push(void 0 === (g = h[x]) ? g : String(g));
          var A = h.groups;
          if (c) {
            var b = [y].concat(E, k, l);
            void 0 !== A && b.push(A);
            var w = String(r.apply(void 0, b));
          } else w = wr(y, l, k, E, A, r);
          k >= m && ((D += l.slice(m, k) + w), (m = k + y.length));
        }
        return D + l.slice(m);
      },
    ];
  });
  var Br = k,
    Tr = Z,
    _r = function (t) {
      if (!Br(t) && null !== t)
        throw TypeError("Can't set " + String(t) + " as a prototype");
      return t;
    },
    zr =
      Object.setPrototypeOf ||
      ("__proto__" in {}
        ? (function () {
            var t,
              e = !1,
              n = {};
            try {
              (t = Object.getOwnPropertyDescriptor(
                Object.prototype,
                "__proto__",
              ).set).call(n, []),
                (e = n instanceof Array);
            } catch (t) {}
            return function (n, r) {
              return Tr(n), _r(r), e ? t.call(n, r) : (n.__proto__ = r), n;
            };
          })()
        : void 0),
    Rr = k,
    Ir = zr,
    Lr = k,
    Or = d,
    $r = Bn("match"),
    Pr = function (t) {
      var e;
      return Lr(t) && (void 0 !== (e = t[$r]) ? !!e : "RegExp" == Or(t));
    },
    Mr = Gt,
    jr = U,
    Nr = u,
    Ur = Bn("species"),
    qr = u,
    Zr = n,
    Hr = Pe,
    Wr = function (t, e, n) {
      var r, i;
      return (
        Ir &&
          "function" == typeof (r = e.constructor) &&
          r !== n &&
          Rr((i = r.prototype)) &&
          i !== n.prototype &&
          Ir(t, i),
        t
      );
    },
    Jr = U.f,
    Vr = Yt.f,
    Kr = Pr,
    Qr = Ve,
    Gr = Ke,
    Yr = X.exports,
    Xr = i,
    ti = Mt.enforce,
    ei = function (t) {
      var e = Mr(t),
        n = jr.f;
      Nr &&
        e &&
        !e[Ur] &&
        n(e, Ur, {
          configurable: !0,
          get: function () {
            return this;
          },
        });
    },
    ni = Bn("match"),
    ri = Zr.RegExp,
    ii = ri.prototype,
    ui = /a/g,
    ai = /a/g,
    oi = new ri(ui) !== ui,
    si = Gr.UNSUPPORTED_Y;
  if (
    qr &&
    Hr(
      "RegExp",
      !oi ||
        si ||
        Xr(function () {
          return (
            (ai[ni] = !1), ri(ui) != ui || ri(ai) == ai || "/a/i" != ri(ui, "i")
          );
        }),
    )
  ) {
    for (
      var li = function (t, e) {
          var n,
            r = this instanceof li,
            i = Kr(t),
            u = void 0 === e;
          if (!r && i && t.constructor === li && u) return t;
          oi
            ? i && !u && (t = t.source)
            : t instanceof li && (u && (e = Qr.call(t)), (t = t.source)),
            si && (n = !!e && e.indexOf("y") > -1) && (e = e.replace(/y/g, ""));
          var a = Wr(oi ? new ri(t, e) : ri(t, e), r ? this : ii, li);
          si && n && (ti(a).sticky = !0);
          return a;
        },
        ci = function (t) {
          (t in li) ||
            Jr(li, t, {
              configurable: !0,
              get: function () {
                return ri[t];
              },
              set: function (e) {
                ri[t] = e;
              },
            });
        },
        pi = Vr(ri),
        di = 0;
      pi.length > di;

    )
      ci(pi[di++]);
    (ii.constructor = li), (li.prototype = ii), Yr(Zr, "RegExp", li);
  }
  ei("RegExp");
  var fi = X.exports,
    hi = Z,
    gi = i,
    Di = Ve,
    mi = "toString",
    vi = RegExp.prototype,
    yi = vi.toString,
    ki = gi(function () {
      return "/a/b" != yi.call({ source: "a", flags: "b" });
    }),
    Ei = yi.name != mi;
  (ki || Ei) &&
    fi(
      RegExp.prototype,
      mi,
      function () {
        var t = hi(this),
          e = String(t.source),
          n = t.flags;
        return (
          "/" +
          e +
          "/" +
          String(
            void 0 === n && t instanceof RegExp && !("flags" in vi)
              ? Di.call(t)
              : n,
          )
        );
      },
      { unsafe: !0 },
    );
  var xi = function (t) {
      if ("function" != typeof t)
        throw TypeError(String(t) + " is not a function");
      return t;
    },
    Ai = Z,
    bi = xi,
    wi = Bn("species"),
    Ci = Un,
    Fi = Pr,
    Si = Z,
    Bi = D,
    Ti = function (t, e) {
      var n,
        r = Ai(t).constructor;
      return void 0 === r || null == (n = Ai(r)[wi]) ? e : bi(n);
    },
    _i = er,
    zi = ie,
    Ri = Wn,
    Ii = sn,
    Li = Ke.UNSUPPORTED_Y,
    Oi = [].push,
    $i = Math.min,
    Pi = 4294967295;
  Ci(
    "split",
    2,
    function (t, e, n) {
      var r;
      return (
        (r =
          "c" == "abbc".split(/(b)*/)[1] ||
          4 != "test".split(/(?:)/, -1).length ||
          2 != "ab".split(/(?:ab)*/).length ||
          4 != ".".split(/(.?)(.?)/).length ||
          ".".split(/()()/).length > 1 ||
          "".split(/.?/).length
            ? function (t, n) {
                var r = String(Bi(this)),
                  i = void 0 === n ? Pi : n >>> 0;
                if (0 === i) return [];
                if (void 0 === t) return [r];
                if (!Fi(t)) return e.call(r, t, i);
                for (
                  var u,
                    a,
                    o,
                    s = [],
                    l =
                      (t.ignoreCase ? "i" : "") +
                      (t.multiline ? "m" : "") +
                      (t.unicode ? "u" : "") +
                      (t.sticky ? "y" : ""),
                    c = 0,
                    p = new RegExp(t.source, l + "g");
                  (u = Ii.call(p, r)) &&
                  !(
                    (a = p.lastIndex) > c &&
                    (s.push(r.slice(c, u.index)),
                    u.length > 1 &&
                      u.index < r.length &&
                      Oi.apply(s, u.slice(1)),
                    (o = u[0].length),
                    (c = a),
                    s.length >= i)
                  );

                )
                  p.lastIndex === u.index && p.lastIndex++;
                return (
                  c === r.length
                    ? (!o && p.test("")) || s.push("")
                    : s.push(r.slice(c)),
                  s.length > i ? s.slice(0, i) : s
                );
              }
            : "0".split(void 0, 0).length
              ? function (t, n) {
                  return void 0 === t && 0 === n ? [] : e.call(this, t, n);
                }
              : e),
        [
          function (e, n) {
            var i = Bi(this),
              u = null == e ? void 0 : e[t];
            return void 0 !== u ? u.call(e, i, n) : r.call(String(i), e, n);
          },
          function (t, i) {
            var u = n(r, t, this, i, r !== e);
            if (u.done) return u.value;
            var a = Si(t),
              o = String(this),
              s = Ti(a, RegExp),
              l = a.unicode,
              c =
                (a.ignoreCase ? "i" : "") +
                (a.multiline ? "m" : "") +
                (a.unicode ? "u" : "") +
                (Li ? "g" : "y"),
              p = new s(Li ? "^(?:" + a.source + ")" : a, c),
              d = void 0 === i ? Pi : i >>> 0;
            if (0 === d) return [];
            if (0 === o.length) return null === Ri(p, o) ? [o] : [];
            for (var f = 0, h = 0, g = []; h < o.length; ) {
              p.lastIndex = Li ? 0 : h;
              var D,
                m = Ri(p, Li ? o.slice(h) : o);
              if (
                null === m ||
                (D = $i(zi(p.lastIndex + (Li ? h : 0)), o.length)) === f
              )
                h = _i(o, h, l);
              else {
                if ((g.push(o.slice(f, h)), g.length === d)) return g;
                for (var v = 1; v <= m.length - 1; v++)
                  if ((g.push(m[v]), g.length === d)) return g;
                h = f = D;
              }
            }
            return g.push(o.slice(f)), g;
          },
        ]
      );
    },
    Li,
  );
  var Mi = "\t\n\v\f\r                　\u2028\u2029\ufeff",
    ji = D,
    Ni = "[\t\n\v\f\r                　\u2028\u2029\ufeff]",
    Ui = RegExp("^" + Ni + Ni + "*"),
    qi = RegExp(Ni + Ni + "*$"),
    Zi = function (t) {
      return function (e) {
        var n = String(ji(e));
        return (
          1 & t && (n = n.replace(Ui, "")), 2 & t && (n = n.replace(qi, "")), n
        );
      };
    },
    Hi = { start: Zi(1), end: Zi(2), trim: Zi(3) },
    Wi = i,
    Ji = Mi,
    Vi = function (t) {
      return Wi(function () {
        return !!Ji[t]() || "​᠎" != "​᠎"[t]() || Ji[t].name !== t;
      });
    },
    Ki = Hi.trim;
  We(
    { target: "String", proto: !0, forced: Vi("trim") },
    {
      trim: function () {
        return Ki(this);
      },
    },
  );
  var Qi = d,
    Gi =
      Array.isArray ||
      function (t) {
        return "Array" == Qi(t);
      },
    Yi = k,
    Xi = Gi,
    tu = Bn("species"),
    eu = function (t, e) {
      var n;
      return (
        Xi(t) &&
          ("function" != typeof (n = t.constructor) ||
          (n !== Array && !Xi(n.prototype))
            ? Yi(n) && null === (n = n[tu]) && (n = void 0)
            : (n = void 0)),
        new (void 0 === n ? Array : n)(0 === e ? 0 : e)
      );
    },
    nu = x,
    ru = U,
    iu = c,
    uu = function (t, e, n) {
      var r = nu(e);
      r in t ? ru.f(t, r, iu(0, n)) : (t[r] = n);
    },
    au = i,
    ou = gn,
    su = Bn("species"),
    lu = function (t) {
      return (
        ou >= 51 ||
        !au(function () {
          var e = [];
          return (
            ((e.constructor = {})[su] = function () {
              return { foo: 1 };
            }),
            1 !== e[t](Boolean).foo
          );
        })
      );
    },
    cu = We,
    pu = se,
    du = ee,
    fu = ie,
    hu = b,
    gu = eu,
    Du = uu,
    mu = lu("splice"),
    vu = Math.max,
    yu = Math.min,
    ku = 9007199254740991,
    Eu = "Maximum allowed length exceeded";
  cu(
    { target: "Array", proto: !0, forced: !mu },
    {
      splice: function (t, e) {
        var n,
          r,
          i,
          u,
          a,
          o,
          s = hu(this),
          l = fu(s.length),
          c = pu(t, l),
          p = arguments.length;
        if (
          (0 === p
            ? (n = r = 0)
            : 1 === p
              ? ((n = 0), (r = l - c))
              : ((n = p - 2), (r = yu(vu(du(e), 0), l - c))),
          l + n - r > ku)
        )
          throw TypeError(Eu);
        for (i = gu(s, r), u = 0; u < r; u++)
          (a = c + u) in s && Du(i, u, s[a]);
        if (((i.length = r), n < r)) {
          for (u = c; u < l - r; u++)
            (o = u + n), (a = u + r) in s ? (s[o] = s[a]) : delete s[o];
          for (u = l; u > l - r + n; u--) delete s[u - 1];
        } else if (n > r)
          for (u = l - r; u > c; u--)
            (o = u + n - 1), (a = u + r - 1) in s ? (s[o] = s[a]) : delete s[o];
        for (u = 0; u < n; u++) s[u + c] = arguments[u + 2];
        return (s.length = l - r + n), i;
      },
    },
  );
  var xu = i,
    Au = function (t, e) {
      var n = [][t];
      return (
        !!n &&
        xu(function () {
          n.call(
            null,
            e ||
              function () {
                throw 1;
              },
            1,
          );
        })
      );
    },
    bu = We,
    wu = y,
    Cu = [].join,
    Fu = g != Object,
    Su = Au("join", ",");
  bu(
    { target: "Array", proto: !0, forced: Fu || !Su },
    {
      join: function (t) {
        return Cu.call(wu(this), void 0 === t ? "," : t);
      },
    },
  );
  var Bu = xi,
    Tu = function (t, e, n) {
      if ((Bu(t), void 0 === e)) return t;
      switch (n) {
        case 0:
          return function () {
            return t.call(e);
          };
        case 1:
          return function (n) {
            return t.call(e, n);
          };
        case 2:
          return function (n, r) {
            return t.call(e, n, r);
          };
        case 3:
          return function (n, r, i) {
            return t.call(e, n, r, i);
          };
      }
      return function () {
        return t.apply(e, arguments);
      };
    },
    _u = g,
    zu = b,
    Ru = ie,
    Iu = eu,
    Lu = [].push,
    Ou = function (t) {
      var e = 1 == t,
        n = 2 == t,
        r = 3 == t,
        i = 4 == t,
        u = 6 == t,
        a = 7 == t,
        o = 5 == t || u;
      return function (s, l, c, p) {
        for (
          var d,
            f,
            h = zu(s),
            g = _u(h),
            D = Tu(l, c, 3),
            m = Ru(g.length),
            v = 0,
            y = p || Iu,
            k = e ? y(s, m) : n || a ? y(s, 0) : void 0;
          m > v;
          v++
        )
          if ((o || v in g) && ((f = D((d = g[v]), v, h)), t))
            if (e) k[v] = f;
            else if (f)
              switch (t) {
                case 3:
                  return !0;
                case 5:
                  return d;
                case 6:
                  return v;
                case 2:
                  Lu.call(k, d);
              }
            else
              switch (t) {
                case 4:
                  return !1;
                case 7:
                  Lu.call(k, d);
              }
        return u ? -1 : r || i ? i : k;
      };
    },
    $u = {
      forEach: Ou(0),
      map: Ou(1),
      filter: Ou(2),
      some: Ou(3),
      every: Ou(4),
      find: Ou(5),
      findIndex: Ou(6),
      filterOut: Ou(7),
    },
    Pu = $u.map;
  We(
    { target: "Array", proto: !0, forced: !lu("map") },
    {
      map: function (t) {
        return Pu(this, t, arguments.length > 1 ? arguments[1] : void 0);
      },
    },
  );
  var Mu = We,
    ju = k,
    Nu = Gi,
    Uu = se,
    qu = ie,
    Zu = y,
    Hu = uu,
    Wu = Bn,
    Ju = lu("slice"),
    Vu = Wu("species"),
    Ku = [].slice,
    Qu = Math.max;
  Mu(
    { target: "Array", proto: !0, forced: !Ju },
    {
      slice: function (t, e) {
        var n,
          r,
          i,
          u = Zu(this),
          a = qu(u.length),
          o = Uu(t, a),
          s = Uu(void 0 === e ? a : e, a);
        if (
          Nu(u) &&
          ("function" != typeof (n = u.constructor) ||
          (n !== Array && !Nu(n.prototype))
            ? ju(n) && null === (n = n[Vu]) && (n = void 0)
            : (n = void 0),
          n === Array || void 0 === n)
        )
          return Ku.call(u, o, s);
        for (
          r = new (void 0 === n ? Array : n)(Qu(s - o, 0)), i = 0;
          o < s;
          o++, i++
        )
          o in u && Hu(r, i, u[o]);
        return (r.length = i), r;
      },
    },
  );
  var Gu = We,
    Yu = Hi.start,
    Xu = Vi("trimStart"),
    ta = Xu
      ? function () {
          return Yu(this);
        }
      : "".trimStart;
  Gu(
    { target: "String", proto: !0, forced: Xu },
    { trimStart: ta, trimLeft: ta },
  );
  var ea = We,
    na = Hi.end,
    ra = Vi("trimEnd"),
    ia = ra
      ? function () {
          return na(this);
        }
      : "".trimEnd;
  ea(
    { target: "String", proto: !0, forced: ra },
    { trimEnd: ia, trimRight: ia },
  );
  var ua = $u.filter;
  We(
    { target: "Array", proto: !0, forced: !lu("filter") },
    {
      filter: function (t) {
        return ua(this, t, arguments.length > 1 ? arguments[1] : void 0);
      },
    },
  );
  var aa = D,
    oa = /"/g,
    sa = i,
    la = function (t, e, n, r) {
      var i = String(aa(t)),
        u = "<" + e;
      return (
        "" !== n &&
          (u += " " + n + '="' + String(r).replace(oa, "&quot;") + '"'),
        u + ">" + i + "</" + e + ">"
      );
    };
  We(
    {
      target: "String",
      proto: !0,
      forced: (function (t) {
        return sa(function () {
          var e = ""[t]('"');
          return e !== e.toLowerCase() || e.split('"').length > 3;
        });
      })("link"),
    },
    {
      link: function (t) {
        return la(this, "a", "href", t);
      },
    },
  );
  var ca = {};
  ca[Bn("toStringTag")] = "z";
  var pa = "[object z]" === String(ca),
    da = pa,
    fa = d,
    ha = Bn("toStringTag"),
    ga =
      "Arguments" ==
      fa(
        (function () {
          return arguments;
        })(),
      ),
    Da = da
      ? fa
      : function (t) {
          var e, n, r;
          return void 0 === t
            ? "Undefined"
            : null === t
              ? "Null"
              : "string" ==
                  typeof (n = (function (t, e) {
                    try {
                      return t[e];
                    } catch (t) {}
                  })((e = Object(t)), ha))
                ? n
                : ga
                  ? fa(e)
                  : "Object" == (r = fa(e)) && "function" == typeof e.callee
                    ? "Arguments"
                    : r;
        },
    ma = pa
      ? {}.toString
      : function () {
          return "[object " + Da(this) + "]";
        },
    va = pa,
    ya = X.exports,
    ka = ma;
  va || ya(Object.prototype, "toString", ka, { unsafe: !0 });
  var Ea = $u.forEach,
    xa = n,
    Aa = {
      CSSRuleList: 0,
      CSSStyleDeclaration: 0,
      CSSValueList: 0,
      ClientRectList: 0,
      DOMRectList: 0,
      DOMStringList: 0,
      DOMTokenList: 1,
      DataTransferItemList: 0,
      FileList: 0,
      HTMLAllCollection: 0,
      HTMLCollection: 0,
      HTMLFormElement: 0,
      HTMLSelectElement: 0,
      MediaList: 0,
      MimeTypeArray: 0,
      NamedNodeMap: 0,
      NodeList: 1,
      PaintRequestList: 0,
      Plugin: 0,
      PluginArray: 0,
      SVGLengthList: 0,
      SVGNumberList: 0,
      SVGPathSegList: 0,
      SVGPointList: 0,
      SVGStringList: 0,
      SVGTransformList: 0,
      SourceBufferList: 0,
      StyleSheetList: 0,
      TextTrackCueList: 0,
      TextTrackList: 0,
      TouchList: 0,
    },
    ba = Au("forEach")
      ? [].forEach
      : function (t) {
          return Ea(this, t, arguments.length > 1 ? arguments[1] : void 0);
        },
    wa = Y;
  for (var Ca in Aa) {
    var Fa = xa[Ca],
      Sa = Fa && Fa.prototype;
    if (Sa && Sa.forEach !== ba)
      try {
        wa(Sa, "forEach", ba);
      } catch (t) {
        Sa.forEach = ba;
      }
  }
  var Ba = ve,
    Ta = ye,
    _a =
      Object.keys ||
      function (t) {
        return Ba(t, Ta);
      },
    za = b,
    Ra = _a;
  We(
    {
      target: "Object",
      stat: !0,
      forced: i(function () {
        Ra(1);
      }),
    },
    {
      keys: function (t) {
        return Ra(za(t));
      },
    },
  );
  var Ia,
    La = U,
    Oa = Z,
    $a = _a,
    Pa = u
      ? Object.defineProperties
      : function (t, e) {
          Oa(t);
          for (var n, r = $a(e), i = r.length, u = 0; i > u; )
            La.f(t, (n = r[u++]), e[n]);
          return t;
        },
    Ma = Gt("document", "documentElement"),
    ja = Z,
    Na = Pa,
    Ua = ye,
    qa = bt,
    Za = Ma,
    Ha = _,
    Wa = At("IE_PROTO"),
    Ja = function () {},
    Va = function (t) {
      return "<script>" + t + "</" + "script>";
    },
    Ka = function () {
      try {
        Ia = document.domain && new ActiveXObject("htmlfile");
      } catch (t) {}
      var t, e;
      Ka = Ia
        ? (function (t) {
            t.write(Va("")), t.close();
            var e = t.parentWindow.Object;
            return (t = null), e;
          })(Ia)
        : (((e = Ha("iframe")).style.display = "none"),
          Za.appendChild(e),
          (e.src = String("javascript:")),
          (t = e.contentWindow.document).open(),
          t.write(Va("document.F=Object")),
          t.close(),
          t.F);
      for (var n = Ua.length; n--; ) delete Ka.prototype[Ua[n]];
      return Ka();
    };
  qa[Wa] = !0;
  var Qa =
      Object.create ||
      function (t, e) {
        var n;
        return (
          null !== t
            ? ((Ja.prototype = ja(t)),
              (n = new Ja()),
              (Ja.prototype = null),
              (n[Wa] = t))
            : (n = Ka()),
          void 0 === e ? n : Na(n, e)
        );
      },
    Ga = U,
    Ya = Bn("unscopables"),
    Xa = Array.prototype;
  null == Xa[Ya] && Ga.f(Xa, Ya, { configurable: !0, value: Qa(null) });
  var to = fe.includes,
    eo = function (t) {
      Xa[Ya][t] = !0;
    };
  We(
    { target: "Array", proto: !0 },
    {
      includes: function (t) {
        return to(this, t, arguments.length > 1 ? arguments[1] : void 0);
      },
    },
  ),
    eo("includes");
  var no = Pr,
    ro = Bn("match"),
    io = function (t) {
      if (no(t))
        throw TypeError("The method doesn't accept regular expressions");
      return t;
    },
    uo = D;
  We(
    {
      target: "String",
      proto: !0,
      forced: !(function (t) {
        var e = /./;
        try {
          "/./"[t](e);
        } catch (n) {
          try {
            return (e[ro] = !1), "/./"[t](e);
          } catch (t) {}
        }
        return !1;
      })("includes"),
    },
    {
      includes: function (t) {
        return !!~String(uo(this)).indexOf(
          io(t),
          arguments.length > 1 ? arguments[1] : void 0,
        );
      },
    },
  );
  var ao = We,
    oo = i,
    so = Gi,
    lo = k,
    co = b,
    po = ie,
    fo = uu,
    ho = eu,
    go = lu,
    Do = gn,
    mo = Bn("isConcatSpreadable"),
    vo = 9007199254740991,
    yo = "Maximum allowed index exceeded",
    ko =
      Do >= 51 ||
      !oo(function () {
        var t = [];
        return (t[mo] = !1), t.concat()[0] !== t;
      }),
    Eo = go("concat"),
    xo = function (t) {
      if (!lo(t)) return !1;
      var e = t[mo];
      return void 0 !== e ? !!e : so(t);
    };
  ao(
    { target: "Array", proto: !0, forced: !ko || !Eo },
    {
      concat: function (t) {
        var e,
          n,
          r,
          i,
          u,
          a = co(this),
          o = ho(a, 0),
          s = 0;
        for (e = -1, r = arguments.length; e < r; e++)
          if (xo((u = -1 === e ? a : arguments[e]))) {
            if (s + (i = po(u.length)) > vo) throw TypeError(yo);
            for (n = 0; n < i; n++, s++) n in u && fo(o, s, u[n]);
          } else {
            if (s >= vo) throw TypeError(yo);
            fo(o, s++, u);
          }
        return (o.length = s), o;
      },
    },
  );
  var Ao = u,
    bo = U.f,
    wo = Function.prototype,
    Co = wo.toString,
    Fo = /^\s*function ([^ (]*)/,
    So = "name";
  function Bo() {
    return {
      baseUrl: null,
      breaks: !1,
      extensions: null,
      gfm: !0,
      headerIds: !0,
      headerPrefix: "",
      highlight: null,
      langPrefix: "language-",
      mangle: !0,
      pedantic: !1,
      renderer: null,
      sanitize: !1,
      sanitizer: null,
      silent: !1,
      smartLists: !1,
      smartypants: !1,
      tokenizer: null,
      walkTokens: null,
      xhtml: !1,
    };
  }
  Ao &&
    !(So in wo) &&
    bo(wo, So, {
      configurable: !0,
      get: function () {
        try {
          return Co.call(this).match(Fo)[1];
        } catch (t) {
          return "";
        }
      },
    });
  var To = {
    baseUrl: null,
    breaks: !1,
    extensions: null,
    gfm: !0,
    headerIds: !0,
    headerPrefix: "",
    highlight: null,
    langPrefix: "language-",
    mangle: !0,
    pedantic: !1,
    renderer: null,
    sanitize: !1,
    sanitizer: null,
    silent: !1,
    smartLists: !1,
    smartypants: !1,
    tokenizer: null,
    walkTokens: null,
    xhtml: !1,
  };
  var _o = /[&<>"']/,
    zo = /[&<>"']/g,
    Ro = /[<>"']|&(?!#?\w+;)/,
    Io = /[<>"']|&(?!#?\w+;)/g,
    Lo = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    },
    Oo = function (t) {
      return Lo[t];
    };
  function $o(t, e) {
    if (e) {
      if (_o.test(t)) return t.replace(zo, Oo);
    } else if (Ro.test(t)) return t.replace(Io, Oo);
    return t;
  }
  var Po = /&(#(?:\d+)|(?:#x[0-9A-Fa-f]+)|(?:\w+));?/gi;
  function Mo(t) {
    return t.replace(Po, function (t, e) {
      return "colon" === (e = e.toLowerCase())
        ? ":"
        : "#" === e.charAt(0)
          ? "x" === e.charAt(1)
            ? String.fromCharCode(parseInt(e.substring(2), 16))
            : String.fromCharCode(+e.substring(1))
          : "";
    });
  }
  var jo = /(^|[^\[])\^/g;
  function No(t, e) {
    (t = t.source || t), (e = e || "");
    var n = {
      replace: function (e, r) {
        return (
          (r = (r = r.source || r).replace(jo, "$1")), (t = t.replace(e, r)), n
        );
      },
      getRegex: function () {
        return new RegExp(t, e);
      },
    };
    return n;
  }
  var Uo = /[^\w:]/g,
    qo = /^$|^[a-z][a-z0-9+.-]*:|^[?#]/i;
  function Zo(t, e, n) {
    if (t) {
      var r;
      try {
        r = decodeURIComponent(Mo(n)).replace(Uo, "").toLowerCase();
      } catch (t) {
        return null;
      }
      if (
        0 === r.indexOf("javascript:") ||
        0 === r.indexOf("vbscript:") ||
        0 === r.indexOf("data:")
      )
        return null;
    }
    e &&
      !qo.test(n) &&
      (n = (function (t, e) {
        Ho[" " + t] ||
          (Wo.test(t)
            ? (Ho[" " + t] = t + "/")
            : (Ho[" " + t] = Yo(t, "/", !0)));
        var n = -1 === (t = Ho[" " + t]).indexOf(":");
        return "//" === e.substring(0, 2)
          ? n
            ? e
            : t.replace(Jo, "$1") + e
          : "/" === e.charAt(0)
            ? n
              ? e
              : t.replace(Vo, "$1") + e
            : t + e;
      })(e, n));
    try {
      n = encodeURI(n).replace(/%25/g, "%");
    } catch (t) {
      return null;
    }
    return n;
  }
  var Ho = {},
    Wo = /^[^:]+:\/*[^/]*$/,
    Jo = /^([^:]+:)[\s\S]*$/,
    Vo = /^([^:]+:\/*[^/]*)[\s\S]*$/;
  var Ko = { exec: function () {} };
  function Qo(t) {
    for (var e, n, r = 1; r < arguments.length; r++)
      for (n in (e = arguments[r]))
        Object.prototype.hasOwnProperty.call(e, n) && (t[n] = e[n]);
    return t;
  }
  function Go(t, e) {
    var n = t
        .replace(/\|/g, function (t, e, n) {
          for (var r = !1, i = e; --i >= 0 && "\\" === n[i]; ) r = !r;
          return r ? "|" : " |";
        })
        .split(/ \|/),
      r = 0;
    if (
      (n[0].trim() || n.shift(),
      n.length > 0 && !n[n.length - 1].trim() && n.pop(),
      n.length > e)
    )
      n.splice(e);
    else for (; n.length < e; ) n.push("");
    for (; r < n.length; r++) n[r] = n[r].trim().replace(/\\\|/g, "|");
    return n;
  }
  function Yo(t, e, n) {
    var r = t.length;
    if (0 === r) return "";
    for (var i = 0; i < r; ) {
      var u = t.charAt(r - i - 1);
      if (u !== e || n) {
        if (u === e || !n) break;
        i++;
      } else i++;
    }
    return t.substr(0, r - i);
  }
  function Xo(t) {
    t &&
      t.sanitize &&
      !t.silent &&
      console.warn(
        "marked(): sanitize and sanitizer parameters are deprecated since version 0.7.0, should not be used and will be removed in the future. Read more here: https://marked.js.org/#/USING_ADVANCED.md#options",
      );
  }
  function ts(t, e) {
    if (e < 1) return "";
    for (var n = ""; e > 1; ) 1 & e && (n += t), (e >>= 1), (t += t);
    return n + t;
  }
  function es(t, e, n, r) {
    var i = e.href,
      u = e.title ? $o(e.title) : null,
      a = t[1].replace(/\\([\[\]])/g, "$1");
    if ("!" !== t[0].charAt(0)) {
      r.state.inLink = !0;
      var o = {
        type: "link",
        raw: n,
        href: i,
        title: u,
        text: a,
        tokens: r.inlineTokens(a, []),
      };
      return (r.state.inLink = !1), o;
    }
    return { type: "image", raw: n, href: i, title: u, text: $o(a) };
  }
  var ns = (function () {
      function t(e) {
        or(this, t), (this.options = e || To);
      }
      return (
        lr(t, [
          {
            key: "space",
            value: function (t) {
              var e = this.rules.block.newline.exec(t);
              if (e && e[0].length > 0) return { type: "space", raw: e[0] };
            },
          },
          {
            key: "code",
            value: function (t) {
              var e = this.rules.block.code.exec(t);
              if (e) {
                var n = e[0].replace(/^ {1,4}/gm, "");
                return {
                  type: "code",
                  raw: e[0],
                  codeBlockStyle: "indented",
                  text: this.options.pedantic ? n : Yo(n, "\n"),
                };
              }
            },
          },
          {
            key: "fences",
            value: function (t) {
              var e = this.rules.block.fences.exec(t);
              if (e) {
                var n = e[0],
                  r = (function (t, e) {
                    var n = t.match(/^(\s+)(?:```)/);
                    if (null === n) return e;
                    var r = n[1];
                    return e
                      .split("\n")
                      .map(function (t) {
                        var e = t.match(/^\s+/);
                        return null === e
                          ? t
                          : cr(e, 1)[0].length >= r.length
                            ? t.slice(r.length)
                            : t;
                      })
                      .join("\n");
                  })(n, e[3] || "");
                return {
                  type: "code",
                  raw: n,
                  lang: e[2] ? e[2].trim() : e[2],
                  text: r,
                };
              }
            },
          },
          {
            key: "heading",
            value: function (t) {
              var e = this.rules.block.heading.exec(t);
              if (e) {
                var n = e[2].trim();
                if (/#$/.test(n)) {
                  var r = Yo(n, "#");
                  this.options.pedantic
                    ? (n = r.trim())
                    : (r && !/ $/.test(r)) || (n = r.trim());
                }
                var i = {
                  type: "heading",
                  raw: e[0],
                  depth: e[1].length,
                  text: n,
                  tokens: [],
                };
                return this.lexer.inline(i.text, i.tokens), i;
              }
            },
          },
          {
            key: "hr",
            value: function (t) {
              var e = this.rules.block.hr.exec(t);
              if (e) return { type: "hr", raw: e[0] };
            },
          },
          {
            key: "blockquote",
            value: function (t) {
              var e = this.rules.block.blockquote.exec(t);
              if (e) {
                var n = e[0].replace(/^ *> ?/gm, "");
                return {
                  type: "blockquote",
                  raw: e[0],
                  tokens: this.lexer.blockTokens(n, []),
                  text: n,
                };
              }
            },
          },
          {
            key: "list",
            value: function (t) {
              var e = this.rules.block.list.exec(t);
              if (e) {
                var n,
                  r,
                  i,
                  u,
                  a,
                  o,
                  s,
                  l,
                  c,
                  p,
                  d,
                  f,
                  h = e[1].trim(),
                  g = h.length > 1,
                  D = {
                    type: "list",
                    raw: "",
                    ordered: g,
                    start: g ? +h.slice(0, -1) : "",
                    loose: !1,
                    items: [],
                  };
                (h = g ? "\\d{1,9}\\".concat(h.slice(-1)) : "\\".concat(h)),
                  this.options.pedantic && (h = g ? h : "[*+-]");
                for (
                  var m = new RegExp(
                    "^( {0,3}".concat(h, ")((?: [^\\n]*)?(?:\\n|$))"),
                  );
                  t &&
                  ((f = !1), (e = m.exec(t))) &&
                  !this.rules.block.hr.test(t);

                ) {
                  if (
                    ((n = e[0]),
                    (t = t.substring(n.length)),
                    (l = e[2].split("\n", 1)[0]),
                    (c = t.split("\n", 1)[0]),
                    this.options.pedantic
                      ? ((u = 2), (d = l.trimLeft()))
                      : ((u = (u = e[2].search(/[^ ]/)) > 4 ? 1 : u),
                        (d = l.slice(u)),
                        (u += e[1].length)),
                    (o = !1),
                    !l &&
                      /^ *$/.test(c) &&
                      ((n += c + "\n"),
                      (t = t.substring(c.length + 1)),
                      (f = !0)),
                    !f)
                  )
                    for (
                      var v = new RegExp(
                        "^ {0,".concat(
                          Math.min(3, u - 1),
                          "}(?:[*+-]|\\d{1,9}[.)])",
                        ),
                      );
                      t &&
                      ((l = p = t.split("\n", 1)[0]),
                      this.options.pedantic &&
                        (l = l.replace(/^ {1,4}(?=( {4})*[^ ])/g, "  ")),
                      !v.test(l));

                    ) {
                      if (l.search(/[^ ]/) >= u || !l.trim())
                        d += "\n" + l.slice(u);
                      else {
                        if (o) break;
                        d += "\n" + l;
                      }
                      o || l.trim() || (o = !0),
                        (n += p + "\n"),
                        (t = t.substring(p.length + 1));
                    }
                  D.loose ||
                    (s ? (D.loose = !0) : /\n *\n *$/.test(n) && (s = !0)),
                    this.options.gfm &&
                      (r = /^\[[ xX]\] /.exec(d)) &&
                      ((i = "[ ] " !== r[0]),
                      (d = d.replace(/^\[[ xX]\] +/, ""))),
                    D.items.push({
                      type: "list_item",
                      raw: n,
                      task: !!r,
                      checked: i,
                      loose: !1,
                      text: d,
                    }),
                    (D.raw += n);
                }
                (D.items[D.items.length - 1].raw = n.trimRight()),
                  (D.items[D.items.length - 1].text = d.trimRight()),
                  (D.raw = D.raw.trimRight());
                var y = D.items.length;
                for (a = 0; a < y; a++) {
                  (this.lexer.state.top = !1),
                    (D.items[a].tokens = this.lexer.blockTokens(
                      D.items[a].text,
                      [],
                    ));
                  var k = D.items[a].tokens.filter(function (t) {
                      return "space" === t.type;
                    }),
                    E = k.every(function (t) {
                      var e,
                        n = 0,
                        r = fr(t.raw.split(""));
                      try {
                        for (r.s(); !(e = r.n()).done; ) {
                          if (("\n" === e.value && (n += 1), n > 1)) return !0;
                        }
                      } catch (t) {
                        r.e(t);
                      } finally {
                        r.f();
                      }
                      return !1;
                    });
                  !D.loose &&
                    k.length &&
                    E &&
                    ((D.loose = !0), (D.items[a].loose = !0));
                }
                return D;
              }
            },
          },
          {
            key: "html",
            value: function (t) {
              var e = this.rules.block.html.exec(t);
              if (e) {
                var n = {
                  type: "html",
                  raw: e[0],
                  pre:
                    !this.options.sanitizer &&
                    ("pre" === e[1] || "script" === e[1] || "style" === e[1]),
                  text: e[0],
                };
                return (
                  this.options.sanitize &&
                    ((n.type = "paragraph"),
                    (n.text = this.options.sanitizer
                      ? this.options.sanitizer(e[0])
                      : $o(e[0])),
                    (n.tokens = []),
                    this.lexer.inline(n.text, n.tokens)),
                  n
                );
              }
            },
          },
          {
            key: "def",
            value: function (t) {
              var e = this.rules.block.def.exec(t);
              if (e)
                return (
                  e[3] && (e[3] = e[3].substring(1, e[3].length - 1)),
                  {
                    type: "def",
                    tag: e[1].toLowerCase().replace(/\s+/g, " "),
                    raw: e[0],
                    href: e[2],
                    title: e[3],
                  }
                );
            },
          },
          {
            key: "table",
            value: function (t) {
              var e = this.rules.block.table.exec(t);
              if (e) {
                var n = {
                  type: "table",
                  header: Go(e[1]).map(function (t) {
                    return { text: t };
                  }),
                  align: e[2].replace(/^ *|\| *$/g, "").split(/ *\| */),
                  rows:
                    e[3] && e[3].trim()
                      ? e[3].replace(/\n[ \t]*$/, "").split("\n")
                      : [],
                };
                if (n.header.length === n.align.length) {
                  n.raw = e[0];
                  var r,
                    i,
                    u,
                    a,
                    o = n.align.length;
                  for (r = 0; r < o; r++)
                    /^ *-+: *$/.test(n.align[r])
                      ? (n.align[r] = "right")
                      : /^ *:-+: *$/.test(n.align[r])
                        ? (n.align[r] = "center")
                        : /^ *:-+ *$/.test(n.align[r])
                          ? (n.align[r] = "left")
                          : (n.align[r] = null);
                  for (o = n.rows.length, r = 0; r < o; r++)
                    n.rows[r] = Go(n.rows[r], n.header.length).map(
                      function (t) {
                        return { text: t };
                      },
                    );
                  for (o = n.header.length, i = 0; i < o; i++)
                    (n.header[i].tokens = []),
                      this.lexer.inlineTokens(
                        n.header[i].text,
                        n.header[i].tokens,
                      );
                  for (o = n.rows.length, i = 0; i < o; i++)
                    for (a = n.rows[i], u = 0; u < a.length; u++)
                      (a[u].tokens = []),
                        this.lexer.inlineTokens(a[u].text, a[u].tokens);
                  return n;
                }
              }
            },
          },
          {
            key: "lheading",
            value: function (t) {
              var e = this.rules.block.lheading.exec(t);
              if (e) {
                var n = {
                  type: "heading",
                  raw: e[0],
                  depth: "=" === e[2].charAt(0) ? 1 : 2,
                  text: e[1],
                  tokens: [],
                };
                return this.lexer.inline(n.text, n.tokens), n;
              }
            },
          },
          {
            key: "paragraph",
            value: function (t) {
              var e = this.rules.block.paragraph.exec(t);
              if (e) {
                var n = {
                  type: "paragraph",
                  raw: e[0],
                  text:
                    "\n" === e[1].charAt(e[1].length - 1)
                      ? e[1].slice(0, -1)
                      : e[1],
                  tokens: [],
                };
                return this.lexer.inline(n.text, n.tokens), n;
              }
            },
          },
          {
            key: "text",
            value: function (t) {
              var e = this.rules.block.text.exec(t);
              if (e) {
                var n = { type: "text", raw: e[0], text: e[0], tokens: [] };
                return this.lexer.inline(n.text, n.tokens), n;
              }
            },
          },
          {
            key: "escape",
            value: function (t) {
              var e = this.rules.inline.escape.exec(t);
              if (e) return { type: "escape", raw: e[0], text: $o(e[1]) };
            },
          },
          {
            key: "tag",
            value: function (t) {
              var e = this.rules.inline.tag.exec(t);
              if (e)
                return (
                  !this.lexer.state.inLink && /^<a /i.test(e[0])
                    ? (this.lexer.state.inLink = !0)
                    : this.lexer.state.inLink &&
                      /^<\/a>/i.test(e[0]) &&
                      (this.lexer.state.inLink = !1),
                  !this.lexer.state.inRawBlock &&
                  /^<(pre|code|kbd|script)(\s|>)/i.test(e[0])
                    ? (this.lexer.state.inRawBlock = !0)
                    : this.lexer.state.inRawBlock &&
                      /^<\/(pre|code|kbd|script)(\s|>)/i.test(e[0]) &&
                      (this.lexer.state.inRawBlock = !1),
                  {
                    type: this.options.sanitize ? "text" : "html",
                    raw: e[0],
                    inLink: this.lexer.state.inLink,
                    inRawBlock: this.lexer.state.inRawBlock,
                    text: this.options.sanitize
                      ? this.options.sanitizer
                        ? this.options.sanitizer(e[0])
                        : $o(e[0])
                      : e[0],
                  }
                );
            },
          },
          {
            key: "link",
            value: function (t) {
              var e = this.rules.inline.link.exec(t);
              if (e) {
                var n = e[2].trim();
                if (!this.options.pedantic && /^</.test(n)) {
                  if (!/>$/.test(n)) return;
                  var r = Yo(n.slice(0, -1), "\\");
                  if ((n.length - r.length) % 2 == 0) return;
                } else {
                  var i = (function (t, e) {
                    if (-1 === t.indexOf(e[1])) return -1;
                    for (var n = t.length, r = 0, i = 0; i < n; i++)
                      if ("\\" === t[i]) i++;
                      else if (t[i] === e[0]) r++;
                      else if (t[i] === e[1] && --r < 0) return i;
                    return -1;
                  })(e[2], "()");
                  if (i > -1) {
                    var u = (0 === e[0].indexOf("!") ? 5 : 4) + e[1].length + i;
                    (e[2] = e[2].substring(0, i)),
                      (e[0] = e[0].substring(0, u).trim()),
                      (e[3] = "");
                  }
                }
                var a = e[2],
                  o = "";
                if (this.options.pedantic) {
                  var s = /^([^'"]*[^\s])\s+(['"])(.*)\2/.exec(a);
                  s && ((a = s[1]), (o = s[3]));
                } else o = e[3] ? e[3].slice(1, -1) : "";
                return (
                  (a = a.trim()),
                  /^</.test(a) &&
                    (a =
                      this.options.pedantic && !/>$/.test(n)
                        ? a.slice(1)
                        : a.slice(1, -1)),
                  es(
                    e,
                    {
                      href: a ? a.replace(this.rules.inline._escapes, "$1") : a,
                      title: o
                        ? o.replace(this.rules.inline._escapes, "$1")
                        : o,
                    },
                    e[0],
                    this.lexer,
                  )
                );
              }
            },
          },
          {
            key: "reflink",
            value: function (t, e) {
              var n;
              if (
                (n = this.rules.inline.reflink.exec(t)) ||
                (n = this.rules.inline.nolink.exec(t))
              ) {
                var r = (n[2] || n[1]).replace(/\s+/g, " ");
                if (!(r = e[r.toLowerCase()]) || !r.href) {
                  var i = n[0].charAt(0);
                  return { type: "text", raw: i, text: i };
                }
                return es(n, r, n[0], this.lexer);
              }
            },
          },
          {
            key: "emStrong",
            value: function (t, e) {
              var n =
                  arguments.length > 2 && void 0 !== arguments[2]
                    ? arguments[2]
                    : "",
                r = this.rules.inline.emStrong.lDelim.exec(t);
              if (
                r &&
                (!r[3] ||
                  !n.match(
                    /(?:[0-9A-Za-z\xAA\xB2\xB3\xB5\xB9\xBA\xBC-\xBE\xC0-\xD6\xD8-\xF6\xF8-\u02C1\u02C6-\u02D1\u02E0-\u02E4\u02EC\u02EE\u0370-\u0374\u0376\u0377\u037A-\u037D\u037F\u0386\u0388-\u038A\u038C\u038E-\u03A1\u03A3-\u03F5\u03F7-\u0481\u048A-\u052F\u0531-\u0556\u0559\u0560-\u0588\u05D0-\u05EA\u05EF-\u05F2\u0620-\u064A\u0660-\u0669\u066E\u066F\u0671-\u06D3\u06D5\u06E5\u06E6\u06EE-\u06FC\u06FF\u0710\u0712-\u072F\u074D-\u07A5\u07B1\u07C0-\u07EA\u07F4\u07F5\u07FA\u0800-\u0815\u081A\u0824\u0828\u0840-\u0858\u0860-\u086A\u08A0-\u08B4\u08B6-\u08C7\u0904-\u0939\u093D\u0950\u0958-\u0961\u0966-\u096F\u0971-\u0980\u0985-\u098C\u098F\u0990\u0993-\u09A8\u09AA-\u09B0\u09B2\u09B6-\u09B9\u09BD\u09CE\u09DC\u09DD\u09DF-\u09E1\u09E6-\u09F1\u09F4-\u09F9\u09FC\u0A05-\u0A0A\u0A0F\u0A10\u0A13-\u0A28\u0A2A-\u0A30\u0A32\u0A33\u0A35\u0A36\u0A38\u0A39\u0A59-\u0A5C\u0A5E\u0A66-\u0A6F\u0A72-\u0A74\u0A85-\u0A8D\u0A8F-\u0A91\u0A93-\u0AA8\u0AAA-\u0AB0\u0AB2\u0AB3\u0AB5-\u0AB9\u0ABD\u0AD0\u0AE0\u0AE1\u0AE6-\u0AEF\u0AF9\u0B05-\u0B0C\u0B0F\u0B10\u0B13-\u0B28\u0B2A-\u0B30\u0B32\u0B33\u0B35-\u0B39\u0B3D\u0B5C\u0B5D\u0B5F-\u0B61\u0B66-\u0B6F\u0B71-\u0B77\u0B83\u0B85-\u0B8A\u0B8E-\u0B90\u0B92-\u0B95\u0B99\u0B9A\u0B9C\u0B9E\u0B9F\u0BA3\u0BA4\u0BA8-\u0BAA\u0BAE-\u0BB9\u0BD0\u0BE6-\u0BF2\u0C05-\u0C0C\u0C0E-\u0C10\u0C12-\u0C28\u0C2A-\u0C39\u0C3D\u0C58-\u0C5A\u0C60\u0C61\u0C66-\u0C6F\u0C78-\u0C7E\u0C80\u0C85-\u0C8C\u0C8E-\u0C90\u0C92-\u0CA8\u0CAA-\u0CB3\u0CB5-\u0CB9\u0CBD\u0CDE\u0CE0\u0CE1\u0CE6-\u0CEF\u0CF1\u0CF2\u0D04-\u0D0C\u0D0E-\u0D10\u0D12-\u0D3A\u0D3D\u0D4E\u0D54-\u0D56\u0D58-\u0D61\u0D66-\u0D78\u0D7A-\u0D7F\u0D85-\u0D96\u0D9A-\u0DB1\u0DB3-\u0DBB\u0DBD\u0DC0-\u0DC6\u0DE6-\u0DEF\u0E01-\u0E30\u0E32\u0E33\u0E40-\u0E46\u0E50-\u0E59\u0E81\u0E82\u0E84\u0E86-\u0E8A\u0E8C-\u0EA3\u0EA5\u0EA7-\u0EB0\u0EB2\u0EB3\u0EBD\u0EC0-\u0EC4\u0EC6\u0ED0-\u0ED9\u0EDC-\u0EDF\u0F00\u0F20-\u0F33\u0F40-\u0F47\u0F49-\u0F6C\u0F88-\u0F8C\u1000-\u102A\u103F-\u1049\u1050-\u1055\u105A-\u105D\u1061\u1065\u1066\u106E-\u1070\u1075-\u1081\u108E\u1090-\u1099\u10A0-\u10C5\u10C7\u10CD\u10D0-\u10FA\u10FC-\u1248\u124A-\u124D\u1250-\u1256\u1258\u125A-\u125D\u1260-\u1288\u128A-\u128D\u1290-\u12B0\u12B2-\u12B5\u12B8-\u12BE\u12C0\u12C2-\u12C5\u12C8-\u12D6\u12D8-\u1310\u1312-\u1315\u1318-\u135A\u1369-\u137C\u1380-\u138F\u13A0-\u13F5\u13F8-\u13FD\u1401-\u166C\u166F-\u167F\u1681-\u169A\u16A0-\u16EA\u16EE-\u16F8\u1700-\u170C\u170E-\u1711\u1720-\u1731\u1740-\u1751\u1760-\u176C\u176E-\u1770\u1780-\u17B3\u17D7\u17DC\u17E0-\u17E9\u17F0-\u17F9\u1810-\u1819\u1820-\u1878\u1880-\u1884\u1887-\u18A8\u18AA\u18B0-\u18F5\u1900-\u191E\u1946-\u196D\u1970-\u1974\u1980-\u19AB\u19B0-\u19C9\u19D0-\u19DA\u1A00-\u1A16\u1A20-\u1A54\u1A80-\u1A89\u1A90-\u1A99\u1AA7\u1B05-\u1B33\u1B45-\u1B4B\u1B50-\u1B59\u1B83-\u1BA0\u1BAE-\u1BE5\u1C00-\u1C23\u1C40-\u1C49\u1C4D-\u1C7D\u1C80-\u1C88\u1C90-\u1CBA\u1CBD-\u1CBF\u1CE9-\u1CEC\u1CEE-\u1CF3\u1CF5\u1CF6\u1CFA\u1D00-\u1DBF\u1E00-\u1F15\u1F18-\u1F1D\u1F20-\u1F45\u1F48-\u1F4D\u1F50-\u1F57\u1F59\u1F5B\u1F5D\u1F5F-\u1F7D\u1F80-\u1FB4\u1FB6-\u1FBC\u1FBE\u1FC2-\u1FC4\u1FC6-\u1FCC\u1FD0-\u1FD3\u1FD6-\u1FDB\u1FE0-\u1FEC\u1FF2-\u1FF4\u1FF6-\u1FFC\u2070\u2071\u2074-\u2079\u207F-\u2089\u2090-\u209C\u2102\u2107\u210A-\u2113\u2115\u2119-\u211D\u2124\u2126\u2128\u212A-\u212D\u212F-\u2139\u213C-\u213F\u2145-\u2149\u214E\u2150-\u2189\u2460-\u249B\u24EA-\u24FF\u2776-\u2793\u2C00-\u2C2E\u2C30-\u2C5E\u2C60-\u2CE4\u2CEB-\u2CEE\u2CF2\u2CF3\u2CFD\u2D00-\u2D25\u2D27\u2D2D\u2D30-\u2D67\u2D6F\u2D80-\u2D96\u2DA0-\u2DA6\u2DA8-\u2DAE\u2DB0-\u2DB6\u2DB8-\u2DBE\u2DC0-\u2DC6\u2DC8-\u2DCE\u2DD0-\u2DD6\u2DD8-\u2DDE\u2E2F\u3005-\u3007\u3021-\u3029\u3031-\u3035\u3038-\u303C\u3041-\u3096\u309D-\u309F\u30A1-\u30FA\u30FC-\u30FF\u3105-\u312F\u3131-\u318E\u3192-\u3195\u31A0-\u31BF\u31F0-\u31FF\u3220-\u3229\u3248-\u324F\u3251-\u325F\u3280-\u3289\u32B1-\u32BF\u3400-\u4DBF\u4E00-\u9FFC\uA000-\uA48C\uA4D0-\uA4FD\uA500-\uA60C\uA610-\uA62B\uA640-\uA66E\uA67F-\uA69D\uA6A0-\uA6EF\uA717-\uA71F\uA722-\uA788\uA78B-\uA7BF\uA7C2-\uA7CA\uA7F5-\uA801\uA803-\uA805\uA807-\uA80A\uA80C-\uA822\uA830-\uA835\uA840-\uA873\uA882-\uA8B3\uA8D0-\uA8D9\uA8F2-\uA8F7\uA8FB\uA8FD\uA8FE\uA900-\uA925\uA930-\uA946\uA960-\uA97C\uA984-\uA9B2\uA9CF-\uA9D9\uA9E0-\uA9E4\uA9E6-\uA9FE\uAA00-\uAA28\uAA40-\uAA42\uAA44-\uAA4B\uAA50-\uAA59\uAA60-\uAA76\uAA7A\uAA7E-\uAAAF\uAAB1\uAAB5\uAAB6\uAAB9-\uAABD\uAAC0\uAAC2\uAADB-\uAADD\uAAE0-\uAAEA\uAAF2-\uAAF4\uAB01-\uAB06\uAB09-\uAB0E\uAB11-\uAB16\uAB20-\uAB26\uAB28-\uAB2E\uAB30-\uAB5A\uAB5C-\uAB69\uAB70-\uABE2\uABF0-\uABF9\uAC00-\uD7A3\uD7B0-\uD7C6\uD7CB-\uD7FB\uF900-\uFA6D\uFA70-\uFAD9\uFB00-\uFB06\uFB13-\uFB17\uFB1D\uFB1F-\uFB28\uFB2A-\uFB36\uFB38-\uFB3C\uFB3E\uFB40\uFB41\uFB43\uFB44\uFB46-\uFBB1\uFBD3-\uFD3D\uFD50-\uFD8F\uFD92-\uFDC7\uFDF0-\uFDFB\uFE70-\uFE74\uFE76-\uFEFC\uFF10-\uFF19\uFF21-\uFF3A\uFF41-\uFF5A\uFF66-\uFFBE\uFFC2-\uFFC7\uFFCA-\uFFCF\uFFD2-\uFFD7\uFFDA-\uFFDC]|\uD800[\uDC00-\uDC0B\uDC0D-\uDC26\uDC28-\uDC3A\uDC3C\uDC3D\uDC3F-\uDC4D\uDC50-\uDC5D\uDC80-\uDCFA\uDD07-\uDD33\uDD40-\uDD78\uDD8A\uDD8B\uDE80-\uDE9C\uDEA0-\uDED0\uDEE1-\uDEFB\uDF00-\uDF23\uDF2D-\uDF4A\uDF50-\uDF75\uDF80-\uDF9D\uDFA0-\uDFC3\uDFC8-\uDFCF\uDFD1-\uDFD5]|\uD801[\uDC00-\uDC9D\uDCA0-\uDCA9\uDCB0-\uDCD3\uDCD8-\uDCFB\uDD00-\uDD27\uDD30-\uDD63\uDE00-\uDF36\uDF40-\uDF55\uDF60-\uDF67]|\uD802[\uDC00-\uDC05\uDC08\uDC0A-\uDC35\uDC37\uDC38\uDC3C\uDC3F-\uDC55\uDC58-\uDC76\uDC79-\uDC9E\uDCA7-\uDCAF\uDCE0-\uDCF2\uDCF4\uDCF5\uDCFB-\uDD1B\uDD20-\uDD39\uDD80-\uDDB7\uDDBC-\uDDCF\uDDD2-\uDE00\uDE10-\uDE13\uDE15-\uDE17\uDE19-\uDE35\uDE40-\uDE48\uDE60-\uDE7E\uDE80-\uDE9F\uDEC0-\uDEC7\uDEC9-\uDEE4\uDEEB-\uDEEF\uDF00-\uDF35\uDF40-\uDF55\uDF58-\uDF72\uDF78-\uDF91\uDFA9-\uDFAF]|\uD803[\uDC00-\uDC48\uDC80-\uDCB2\uDCC0-\uDCF2\uDCFA-\uDD23\uDD30-\uDD39\uDE60-\uDE7E\uDE80-\uDEA9\uDEB0\uDEB1\uDF00-\uDF27\uDF30-\uDF45\uDF51-\uDF54\uDFB0-\uDFCB\uDFE0-\uDFF6]|\uD804[\uDC03-\uDC37\uDC52-\uDC6F\uDC83-\uDCAF\uDCD0-\uDCE8\uDCF0-\uDCF9\uDD03-\uDD26\uDD36-\uDD3F\uDD44\uDD47\uDD50-\uDD72\uDD76\uDD83-\uDDB2\uDDC1-\uDDC4\uDDD0-\uDDDA\uDDDC\uDDE1-\uDDF4\uDE00-\uDE11\uDE13-\uDE2B\uDE80-\uDE86\uDE88\uDE8A-\uDE8D\uDE8F-\uDE9D\uDE9F-\uDEA8\uDEB0-\uDEDE\uDEF0-\uDEF9\uDF05-\uDF0C\uDF0F\uDF10\uDF13-\uDF28\uDF2A-\uDF30\uDF32\uDF33\uDF35-\uDF39\uDF3D\uDF50\uDF5D-\uDF61]|\uD805[\uDC00-\uDC34\uDC47-\uDC4A\uDC50-\uDC59\uDC5F-\uDC61\uDC80-\uDCAF\uDCC4\uDCC5\uDCC7\uDCD0-\uDCD9\uDD80-\uDDAE\uDDD8-\uDDDB\uDE00-\uDE2F\uDE44\uDE50-\uDE59\uDE80-\uDEAA\uDEB8\uDEC0-\uDEC9\uDF00-\uDF1A\uDF30-\uDF3B]|\uD806[\uDC00-\uDC2B\uDCA0-\uDCF2\uDCFF-\uDD06\uDD09\uDD0C-\uDD13\uDD15\uDD16\uDD18-\uDD2F\uDD3F\uDD41\uDD50-\uDD59\uDDA0-\uDDA7\uDDAA-\uDDD0\uDDE1\uDDE3\uDE00\uDE0B-\uDE32\uDE3A\uDE50\uDE5C-\uDE89\uDE9D\uDEC0-\uDEF8]|\uD807[\uDC00-\uDC08\uDC0A-\uDC2E\uDC40\uDC50-\uDC6C\uDC72-\uDC8F\uDD00-\uDD06\uDD08\uDD09\uDD0B-\uDD30\uDD46\uDD50-\uDD59\uDD60-\uDD65\uDD67\uDD68\uDD6A-\uDD89\uDD98\uDDA0-\uDDA9\uDEE0-\uDEF2\uDFB0\uDFC0-\uDFD4]|\uD808[\uDC00-\uDF99]|\uD809[\uDC00-\uDC6E\uDC80-\uDD43]|[\uD80C\uD81C-\uD820\uD822\uD840-\uD868\uD86A-\uD86C\uD86F-\uD872\uD874-\uD879\uD880-\uD883][\uDC00-\uDFFF]|\uD80D[\uDC00-\uDC2E]|\uD811[\uDC00-\uDE46]|\uD81A[\uDC00-\uDE38\uDE40-\uDE5E\uDE60-\uDE69\uDED0-\uDEED\uDF00-\uDF2F\uDF40-\uDF43\uDF50-\uDF59\uDF5B-\uDF61\uDF63-\uDF77\uDF7D-\uDF8F]|\uD81B[\uDE40-\uDE96\uDF00-\uDF4A\uDF50\uDF93-\uDF9F\uDFE0\uDFE1\uDFE3]|\uD821[\uDC00-\uDFF7]|\uD823[\uDC00-\uDCD5\uDD00-\uDD08]|\uD82C[\uDC00-\uDD1E\uDD50-\uDD52\uDD64-\uDD67\uDD70-\uDEFB]|\uD82F[\uDC00-\uDC6A\uDC70-\uDC7C\uDC80-\uDC88\uDC90-\uDC99]|\uD834[\uDEE0-\uDEF3\uDF60-\uDF78]|\uD835[\uDC00-\uDC54\uDC56-\uDC9C\uDC9E\uDC9F\uDCA2\uDCA5\uDCA6\uDCA9-\uDCAC\uDCAE-\uDCB9\uDCBB\uDCBD-\uDCC3\uDCC5-\uDD05\uDD07-\uDD0A\uDD0D-\uDD14\uDD16-\uDD1C\uDD1E-\uDD39\uDD3B-\uDD3E\uDD40-\uDD44\uDD46\uDD4A-\uDD50\uDD52-\uDEA5\uDEA8-\uDEC0\uDEC2-\uDEDA\uDEDC-\uDEFA\uDEFC-\uDF14\uDF16-\uDF34\uDF36-\uDF4E\uDF50-\uDF6E\uDF70-\uDF88\uDF8A-\uDFA8\uDFAA-\uDFC2\uDFC4-\uDFCB\uDFCE-\uDFFF]|\uD838[\uDD00-\uDD2C\uDD37-\uDD3D\uDD40-\uDD49\uDD4E\uDEC0-\uDEEB\uDEF0-\uDEF9]|\uD83A[\uDC00-\uDCC4\uDCC7-\uDCCF\uDD00-\uDD43\uDD4B\uDD50-\uDD59]|\uD83B[\uDC71-\uDCAB\uDCAD-\uDCAF\uDCB1-\uDCB4\uDD01-\uDD2D\uDD2F-\uDD3D\uDE00-\uDE03\uDE05-\uDE1F\uDE21\uDE22\uDE24\uDE27\uDE29-\uDE32\uDE34-\uDE37\uDE39\uDE3B\uDE42\uDE47\uDE49\uDE4B\uDE4D-\uDE4F\uDE51\uDE52\uDE54\uDE57\uDE59\uDE5B\uDE5D\uDE5F\uDE61\uDE62\uDE64\uDE67-\uDE6A\uDE6C-\uDE72\uDE74-\uDE77\uDE79-\uDE7C\uDE7E\uDE80-\uDE89\uDE8B-\uDE9B\uDEA1-\uDEA3\uDEA5-\uDEA9\uDEAB-\uDEBB]|\uD83C[\uDD00-\uDD0C]|\uD83E[\uDFF0-\uDFF9]|\uD869[\uDC00-\uDEDD\uDF00-\uDFFF]|\uD86D[\uDC00-\uDF34\uDF40-\uDFFF]|\uD86E[\uDC00-\uDC1D\uDC20-\uDFFF]|\uD873[\uDC00-\uDEA1\uDEB0-\uDFFF]|\uD87A[\uDC00-\uDFE0]|\uD87E[\uDC00-\uDE1D]|\uD884[\uDC00-\uDF4A])/,
                  ))
              ) {
                var i = r[1] || r[2] || "";
                if (
                  !i ||
                  (i && ("" === n || this.rules.inline.punctuation.exec(n)))
                ) {
                  var u,
                    a,
                    o = r[0].length - 1,
                    s = o,
                    l = 0,
                    c =
                      "*" === r[0][0]
                        ? this.rules.inline.emStrong.rDelimAst
                        : this.rules.inline.emStrong.rDelimUnd;
                  for (
                    c.lastIndex = 0, e = e.slice(-1 * t.length + o);
                    null != (r = c.exec(e));

                  )
                    if ((u = r[1] || r[2] || r[3] || r[4] || r[5] || r[6]))
                      if (((a = u.length), r[3] || r[4])) s += a;
                      else if (!((r[5] || r[6]) && o % 3) || (o + a) % 3) {
                        if (!((s -= a) > 0)) {
                          if (
                            ((a = Math.min(a, a + s + l)), Math.min(o, a) % 2)
                          ) {
                            var p = t.slice(1, o + r.index + a);
                            return {
                              type: "em",
                              raw: t.slice(0, o + r.index + a + 1),
                              text: p,
                              tokens: this.lexer.inlineTokens(p, []),
                            };
                          }
                          var d = t.slice(2, o + r.index + a - 1);
                          return {
                            type: "strong",
                            raw: t.slice(0, o + r.index + a + 1),
                            text: d,
                            tokens: this.lexer.inlineTokens(d, []),
                          };
                        }
                      } else l += a;
                }
              }
            },
          },
          {
            key: "codespan",
            value: function (t) {
              var e = this.rules.inline.code.exec(t);
              if (e) {
                var n = e[2].replace(/\n/g, " "),
                  r = /[^ ]/.test(n),
                  i = /^ /.test(n) && / $/.test(n);
                return (
                  r && i && (n = n.substring(1, n.length - 1)),
                  (n = $o(n, !0)),
                  { type: "codespan", raw: e[0], text: n }
                );
              }
            },
          },
          {
            key: "br",
            value: function (t) {
              var e = this.rules.inline.br.exec(t);
              if (e) return { type: "br", raw: e[0] };
            },
          },
          {
            key: "del",
            value: function (t) {
              var e = this.rules.inline.del.exec(t);
              if (e)
                return {
                  type: "del",
                  raw: e[0],
                  text: e[2],
                  tokens: this.lexer.inlineTokens(e[2], []),
                };
            },
          },
          {
            key: "autolink",
            value: function (t, e) {
              var n,
                r,
                i = this.rules.inline.autolink.exec(t);
              if (i)
                return (
                  (r =
                    "@" === i[2]
                      ? "mailto:" +
                        (n = $o(this.options.mangle ? e(i[1]) : i[1]))
                      : (n = $o(i[1]))),
                  {
                    type: "link",
                    raw: i[0],
                    text: n,
                    href: r,
                    tokens: [{ type: "text", raw: n, text: n }],
                  }
                );
            },
          },
          {
            key: "url",
            value: function (t, e) {
              var n;
              if ((n = this.rules.inline.url.exec(t))) {
                var r, i;
                if ("@" === n[2])
                  i =
                    "mailto:" + (r = $o(this.options.mangle ? e(n[0]) : n[0]));
                else {
                  var u;
                  do {
                    (u = n[0]),
                      (n[0] = this.rules.inline._backpedal.exec(n[0])[0]);
                  } while (u !== n[0]);
                  (r = $o(n[0])), (i = "www." === n[1] ? "http://" + r : r);
                }
                return {
                  type: "link",
                  raw: n[0],
                  text: r,
                  href: i,
                  tokens: [{ type: "text", raw: r, text: r }],
                };
              }
            },
          },
          {
            key: "inlineText",
            value: function (t, e) {
              var n,
                r = this.rules.inline.text.exec(t);
              if (r)
                return (
                  (n = this.lexer.state.inRawBlock
                    ? this.options.sanitize
                      ? this.options.sanitizer
                        ? this.options.sanitizer(r[0])
                        : $o(r[0])
                      : r[0]
                    : $o(this.options.smartypants ? e(r[0]) : r[0])),
                  { type: "text", raw: r[0], text: n }
                );
            },
          },
        ]),
        t
      );
    })(),
    rs = {
      newline: /^(?: *(?:\n|$))+/,
      code: /^( {4}[^\n]+(?:\n(?: *(?:\n|$))*)?)+/,
      fences:
        /^ {0,3}(`{3,}(?=[^`\n]*\n)|~{3,})([^\n]*)\n(?:|([\s\S]*?)\n)(?: {0,3}\1[~`]* *(?=\n|$)|$)/,
      hr: /^ {0,3}((?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$)/,
      heading: /^ {0,3}(#{1,6})(?=\s|$)(.*)(?:\n+|$)/,
      blockquote: /^( {0,3}> ?(paragraph|[^\n]*)(?:\n|$))+/,
      list: /^( {0,3}bull)( [^\n]+?)?(?:\n|$)/,
      html: "^ {0,3}(?:<(script|pre|style|textarea)[\\s>][\\s\\S]*?(?:</\\1>[^\\n]*\\n+|$)|comment[^\\n]*(\\n+|$)|<\\?[\\s\\S]*?(?:\\?>\\n*|$)|<![A-Z][\\s\\S]*?(?:>\\n*|$)|<!\\[CDATA\\[[\\s\\S]*?(?:\\]\\]>\\n*|$)|</?(tag)(?: +|\\n|/?>)[\\s\\S]*?(?:(?:\\n *)+\\n|$)|<(?!script|pre|style|textarea)([a-z][\\w-]*)(?:attribute)*? */?>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n *)+\\n|$)|</(?!script|pre|style|textarea)[a-z][\\w-]*\\s*>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n *)+\\n|$))",
      def: /^ {0,3}\[(label)\]: *(?:\n *)?<?([^\s>]+)>?(?:(?: +(?:\n *)?| *\n *)(title))? *(?:\n+|$)/,
      table: Ko,
      lheading: /^([^\n]+)\n {0,3}(=+|-+) *(?:\n+|$)/,
      _paragraph:
        /^([^\n]+(?:\n(?!hr|heading|lheading|blockquote|fences|list|html|table| +\n)[^\n]+)*)/,
      text: /^[^\n]+/,
      _label: /(?!\s*\])(?:\\.|[^\[\]\\])+/,
      _title: /(?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))/,
    };
  (rs.def = No(rs.def)
    .replace("label", rs._label)
    .replace("title", rs._title)
    .getRegex()),
    (rs.bullet = /(?:[*+-]|\d{1,9}[.)])/),
    (rs.listItemStart = No(/^( *)(bull) */)
      .replace("bull", rs.bullet)
      .getRegex()),
    (rs.list = No(rs.list)
      .replace(/bull/g, rs.bullet)
      .replace(
        "hr",
        "\\n+(?=\\1?(?:(?:- *){3,}|(?:_ *){3,}|(?:\\* *){3,})(?:\\n+|$))",
      )
      .replace("def", "\\n+(?=" + rs.def.source + ")")
      .getRegex()),
    (rs._tag =
      "address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul"),
    (rs._comment = /<!--(?!-?>)[\s\S]*?(?:-->|$)/),
    (rs.html = No(rs.html, "i")
      .replace("comment", rs._comment)
      .replace("tag", rs._tag)
      .replace(
        "attribute",
        / +[a-zA-Z:_][\w.:-]*(?: *= *"[^"\n]*"| *= *'[^'\n]*'| *= *[^\s"'=<>`]+)?/,
      )
      .getRegex()),
    (rs.paragraph = No(rs._paragraph)
      .replace("hr", rs.hr)
      .replace("heading", " {0,3}#{1,6} ")
      .replace("|lheading", "")
      .replace("|table", "")
      .replace("blockquote", " {0,3}>")
      .replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n")
      .replace("list", " {0,3}(?:[*+-]|1[.)]) ")
      .replace(
        "html",
        "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)",
      )
      .replace("tag", rs._tag)
      .getRegex()),
    (rs.blockquote = No(rs.blockquote)
      .replace("paragraph", rs.paragraph)
      .getRegex()),
    (rs.normal = Qo({}, rs)),
    (rs.gfm = Qo({}, rs.normal, {
      table:
        "^ *([^\\n ].*\\|.*)\\n {0,3}(?:\\| *)?(:?-+:? *(?:\\| *:?-+:? *)*)(?:\\| *)?(?:\\n((?:(?! *\\n|hr|heading|blockquote|code|fences|list|html).*(?:\\n|$))*)\\n*|$)",
    })),
    (rs.gfm.table = No(rs.gfm.table)
      .replace("hr", rs.hr)
      .replace("heading", " {0,3}#{1,6} ")
      .replace("blockquote", " {0,3}>")
      .replace("code", " {4}[^\\n]")
      .replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n")
      .replace("list", " {0,3}(?:[*+-]|1[.)]) ")
      .replace(
        "html",
        "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)",
      )
      .replace("tag", rs._tag)
      .getRegex()),
    (rs.gfm.paragraph = No(rs._paragraph)
      .replace("hr", rs.hr)
      .replace("heading", " {0,3}#{1,6} ")
      .replace("|lheading", "")
      .replace("table", rs.gfm.table)
      .replace("blockquote", " {0,3}>")
      .replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n")
      .replace("list", " {0,3}(?:[*+-]|1[.)]) ")
      .replace(
        "html",
        "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)",
      )
      .replace("tag", rs._tag)
      .getRegex()),
    (rs.pedantic = Qo({}, rs.normal, {
      html: No(
        "^ *(?:comment *(?:\\n|\\s*$)|<(tag)[\\s\\S]+?</\\1> *(?:\\n{2,}|\\s*$)|<tag(?:\"[^\"]*\"|'[^']*'|\\s[^'\"/>\\s]*)*?/?> *(?:\\n{2,}|\\s*$))",
      )
        .replace("comment", rs._comment)
        .replace(
          /tag/g,
          "(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\b)\\w+(?!:|[^\\w\\s@]*@)\\b",
        )
        .getRegex(),
      def: /^ *\[([^\]]+)\]: *<?([^\s>]+)>?(?: +(["(][^\n]+[")]))? *(?:\n+|$)/,
      heading: /^(#{1,6})(.*)(?:\n+|$)/,
      fences: Ko,
      paragraph: No(rs.normal._paragraph)
        .replace("hr", rs.hr)
        .replace("heading", " *#{1,6} *[^\n]")
        .replace("lheading", rs.lheading)
        .replace("blockquote", " {0,3}>")
        .replace("|fences", "")
        .replace("|list", "")
        .replace("|html", "")
        .getRegex(),
    }));
  var is = {
    escape: /^\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/,
    autolink: /^<(scheme:[^\s\x00-\x1f<>]*|email)>/,
    url: Ko,
    tag: "^comment|^</[a-zA-Z][\\w:-]*\\s*>|^<[a-zA-Z][\\w-]*(?:attribute)*?\\s*/?>|^<\\?[\\s\\S]*?\\?>|^<![a-zA-Z]+\\s[\\s\\S]*?>|^<!\\[CDATA\\[[\\s\\S]*?\\]\\]>",
    link: /^!?\[(label)\]\(\s*(href)(?:\s+(title))?\s*\)/,
    reflink: /^!?\[(label)\]\[(ref)\]/,
    nolink: /^!?\[(ref)\](?:\[\])?/,
    reflinkSearch: "reflink|nolink(?!\\()",
    emStrong: {
      lDelim: /^(?:\*+(?:([punct_])|[^\s*]))|^_+(?:([punct*])|([^\s_]))/,
      rDelimAst:
        /^[^_*]*?\_\_[^_*]*?\*[^_*]*?(?=\_\_)|[punct_](\*+)(?=[\s]|$)|[^punct*_\s](\*+)(?=[punct_\s]|$)|[punct_\s](\*+)(?=[^punct*_\s])|[\s](\*+)(?=[punct_])|[punct_](\*+)(?=[punct_])|[^punct*_\s](\*+)(?=[^punct*_\s])/,
      rDelimUnd:
        /^[^_*]*?\*\*[^_*]*?\_[^_*]*?(?=\*\*)|[punct*](\_+)(?=[\s]|$)|[^punct*_\s](\_+)(?=[punct*\s]|$)|[punct*\s](\_+)(?=[^punct*_\s])|[\s](\_+)(?=[punct*])|[punct*](\_+)(?=[punct*])/,
    },
    code: /^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/,
    br: /^( {2,}|\\)\n(?!\s*$)/,
    del: Ko,
    text: /^(`+|[^`])(?:(?= {2,}\n)|[\s\S]*?(?:(?=[\\<!\[`*_]|\b_|$)|[^ ](?= {2,}\n)))/,
    punctuation: /^([\spunctuation])/,
  };
  function us(t) {
    return t
      .replace(/---/g, "—")
      .replace(/--/g, "–")
      .replace(/(^|[-\u2014/(\[{"\s])'/g, "$1‘")
      .replace(/'/g, "’")
      .replace(/(^|[-\u2014/(\[{\u2018\s])"/g, "$1“")
      .replace(/"/g, "”")
      .replace(/\.{3}/g, "…");
  }
  function as(t) {
    var e,
      n,
      r = "",
      i = t.length;
    for (e = 0; e < i; e++)
      (n = t.charCodeAt(e)),
        Math.random() > 0.5 && (n = "x" + n.toString(16)),
        (r += "&#" + n + ";");
    return r;
  }
  (is._punctuation = "!\"#$%&'()+\\-.,/:;<=>?@\\[\\]`^{|}~"),
    (is.punctuation = No(is.punctuation)
      .replace(/punctuation/g, is._punctuation)
      .getRegex()),
    (is.blockSkip = /\[[^\]]*?\]\([^\)]*?\)|`[^`]*?`|<[^>]*?>/g),
    (is.escapedEmSt = /\\\*|\\_/g),
    (is._comment = No(rs._comment)
      .replace("(?:--\x3e|$)", "--\x3e")
      .getRegex()),
    (is.emStrong.lDelim = No(is.emStrong.lDelim)
      .replace(/punct/g, is._punctuation)
      .getRegex()),
    (is.emStrong.rDelimAst = No(is.emStrong.rDelimAst, "g")
      .replace(/punct/g, is._punctuation)
      .getRegex()),
    (is.emStrong.rDelimUnd = No(is.emStrong.rDelimUnd, "g")
      .replace(/punct/g, is._punctuation)
      .getRegex()),
    (is._escapes = /\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/g),
    (is._scheme = /[a-zA-Z][a-zA-Z0-9+.-]{1,31}/),
    (is._email =
      /[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])/),
    (is.autolink = No(is.autolink)
      .replace("scheme", is._scheme)
      .replace("email", is._email)
      .getRegex()),
    (is._attribute =
      /\s+[a-zA-Z:_][\w.:-]*(?:\s*=\s*"[^"]*"|\s*=\s*'[^']*'|\s*=\s*[^\s"'=<>`]+)?/),
    (is.tag = No(is.tag)
      .replace("comment", is._comment)
      .replace("attribute", is._attribute)
      .getRegex()),
    (is._label = /(?:\[(?:\\.|[^\[\]\\])*\]|\\.|`[^`]*`|[^\[\]\\`])*?/),
    (is._href = /<(?:\\.|[^\n<>\\])+>|[^\s\x00-\x1f]*/),
    (is._title = /"(?:\\"?|[^"\\])*"|'(?:\\'?|[^'\\])*'|\((?:\\\)?|[^)\\])*\)/),
    (is.link = No(is.link)
      .replace("label", is._label)
      .replace("href", is._href)
      .replace("title", is._title)
      .getRegex()),
    (is.reflink = No(is.reflink)
      .replace("label", is._label)
      .replace("ref", rs._label)
      .getRegex()),
    (is.nolink = No(is.nolink).replace("ref", rs._label).getRegex()),
    (is.reflinkSearch = No(is.reflinkSearch, "g")
      .replace("reflink", is.reflink)
      .replace("nolink", is.nolink)
      .getRegex()),
    (is.normal = Qo({}, is)),
    (is.pedantic = Qo({}, is.normal, {
      strong: {
        start: /^__|\*\*/,
        middle:
          /^__(?=\S)([\s\S]*?\S)__(?!_)|^\*\*(?=\S)([\s\S]*?\S)\*\*(?!\*)/,
        endAst: /\*\*(?!\*)/g,
        endUnd: /__(?!_)/g,
      },
      em: {
        start: /^_|\*/,
        middle: /^()\*(?=\S)([\s\S]*?\S)\*(?!\*)|^_(?=\S)([\s\S]*?\S)_(?!_)/,
        endAst: /\*(?!\*)/g,
        endUnd: /_(?!_)/g,
      },
      link: No(/^!?\[(label)\]\((.*?)\)/)
        .replace("label", is._label)
        .getRegex(),
      reflink: No(/^!?\[(label)\]\s*\[([^\]]*)\]/)
        .replace("label", is._label)
        .getRegex(),
    })),
    (is.gfm = Qo({}, is.normal, {
      escape: No(is.escape).replace("])", "~|])").getRegex(),
      _extended_email:
        /[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/,
      url: /^((?:ftp|https?):\/\/|www\.)(?:[a-zA-Z0-9\-]+\.?)+[^\s<]*|^email/,
      _backpedal:
        /(?:[^?!.,:;*_~()&]+|\([^)]*\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_~)]+(?!$))+/,
      del: /^(~~?)(?=[^\s~])([\s\S]*?[^\s~])\1(?=[^~]|$)/,
      text: /^([`~]+|[^`~])(?:(?= {2,}\n)|(?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)|[\s\S]*?(?:(?=[\\<!\[`*~_]|\b_|https?:\/\/|ftp:\/\/|www\.|$)|[^ ](?= {2,}\n)|[^a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-](?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)))/,
    })),
    (is.gfm.url = No(is.gfm.url, "i")
      .replace("email", is.gfm._extended_email)
      .getRegex()),
    (is.breaks = Qo({}, is.gfm, {
      br: No(is.br).replace("{2,}", "*").getRegex(),
      text: No(is.gfm.text)
        .replace("\\b_", "\\b_| {2,}\\n")
        .replace(/\{2,\}/g, "*")
        .getRegex(),
    }));
  var os = (function () {
      function t(e) {
        or(this, t),
          (this.tokens = []),
          (this.tokens.links = Object.create(null)),
          (this.options = e || To),
          (this.options.tokenizer = this.options.tokenizer || new ns()),
          (this.tokenizer = this.options.tokenizer),
          (this.tokenizer.options = this.options),
          (this.tokenizer.lexer = this),
          (this.inlineQueue = []),
          (this.state = { inLink: !1, inRawBlock: !1, top: !0 });
        var n = { block: rs.normal, inline: is.normal };
        this.options.pedantic
          ? ((n.block = rs.pedantic), (n.inline = is.pedantic))
          : this.options.gfm &&
            ((n.block = rs.gfm),
            this.options.breaks ? (n.inline = is.breaks) : (n.inline = is.gfm)),
          (this.tokenizer.rules = n);
      }
      return (
        lr(
          t,
          [
            {
              key: "lex",
              value: function (t) {
                var e;
                for (
                  t = t.replace(/\r\n|\r/g, "\n").replace(/\t/g, "    "),
                    this.blockTokens(t, this.tokens);
                  (e = this.inlineQueue.shift());

                )
                  this.inlineTokens(e.src, e.tokens);
                return this.tokens;
              },
            },
            {
              key: "blockTokens",
              value: function (t) {
                var e,
                  n,
                  r,
                  i,
                  u = this,
                  a =
                    arguments.length > 1 && void 0 !== arguments[1]
                      ? arguments[1]
                      : [];
                for (
                  this.options.pedantic && (t = t.replace(/^ +$/gm, ""));
                  t;

                )
                  if (
                    !(
                      this.options.extensions &&
                      this.options.extensions.block &&
                      this.options.extensions.block.some(function (n) {
                        return (
                          !!(e = n.call({ lexer: u }, t, a)) &&
                          ((t = t.substring(e.raw.length)), a.push(e), !0)
                        );
                      })
                    )
                  )
                    if ((e = this.tokenizer.space(t)))
                      (t = t.substring(e.raw.length)),
                        1 === e.raw.length && a.length > 0
                          ? (a[a.length - 1].raw += "\n")
                          : a.push(e);
                    else if ((e = this.tokenizer.code(t)))
                      (t = t.substring(e.raw.length)),
                        !(n = a[a.length - 1]) ||
                        ("paragraph" !== n.type && "text" !== n.type)
                          ? a.push(e)
                          : ((n.raw += "\n" + e.raw),
                            (n.text += "\n" + e.text),
                            (this.inlineQueue[this.inlineQueue.length - 1].src =
                              n.text));
                    else if ((e = this.tokenizer.fences(t)))
                      (t = t.substring(e.raw.length)), a.push(e);
                    else if ((e = this.tokenizer.heading(t)))
                      (t = t.substring(e.raw.length)), a.push(e);
                    else if ((e = this.tokenizer.hr(t)))
                      (t = t.substring(e.raw.length)), a.push(e);
                    else if ((e = this.tokenizer.blockquote(t)))
                      (t = t.substring(e.raw.length)), a.push(e);
                    else if ((e = this.tokenizer.list(t)))
                      (t = t.substring(e.raw.length)), a.push(e);
                    else if ((e = this.tokenizer.html(t)))
                      (t = t.substring(e.raw.length)), a.push(e);
                    else if ((e = this.tokenizer.def(t)))
                      (t = t.substring(e.raw.length)),
                        !(n = a[a.length - 1]) ||
                        ("paragraph" !== n.type && "text" !== n.type)
                          ? this.tokens.links[e.tag] ||
                            (this.tokens.links[e.tag] = {
                              href: e.href,
                              title: e.title,
                            })
                          : ((n.raw += "\n" + e.raw),
                            (n.text += "\n" + e.raw),
                            (this.inlineQueue[this.inlineQueue.length - 1].src =
                              n.text));
                    else if ((e = this.tokenizer.table(t)))
                      (t = t.substring(e.raw.length)), a.push(e);
                    else if ((e = this.tokenizer.lheading(t)))
                      (t = t.substring(e.raw.length)), a.push(e);
                    else if (
                      ((r = t),
                      this.options.extensions &&
                        this.options.extensions.startBlock &&
                        (function () {
                          var e = 1 / 0,
                            n = t.slice(1),
                            i = void 0;
                          u.options.extensions.startBlock.forEach(function (t) {
                            "number" ==
                              typeof (i = t.call({ lexer: this }, n)) &&
                              i >= 0 &&
                              (e = Math.min(e, i));
                          }),
                            e < 1 / 0 && e >= 0 && (r = t.substring(0, e + 1));
                        })(),
                      this.state.top && (e = this.tokenizer.paragraph(r)))
                    )
                      (n = a[a.length - 1]),
                        i && "paragraph" === n.type
                          ? ((n.raw += "\n" + e.raw),
                            (n.text += "\n" + e.text),
                            this.inlineQueue.pop(),
                            (this.inlineQueue[this.inlineQueue.length - 1].src =
                              n.text))
                          : a.push(e),
                        (i = r.length !== t.length),
                        (t = t.substring(e.raw.length));
                    else if ((e = this.tokenizer.text(t)))
                      (t = t.substring(e.raw.length)),
                        (n = a[a.length - 1]) && "text" === n.type
                          ? ((n.raw += "\n" + e.raw),
                            (n.text += "\n" + e.text),
                            this.inlineQueue.pop(),
                            (this.inlineQueue[this.inlineQueue.length - 1].src =
                              n.text))
                          : a.push(e);
                    else if (t) {
                      var o = "Infinite loop on byte: " + t.charCodeAt(0);
                      if (this.options.silent) {
                        console.error(o);
                        break;
                      }
                      throw new Error(o);
                    }
                return (this.state.top = !0), a;
              },
            },
            {
              key: "inline",
              value: function (t, e) {
                this.inlineQueue.push({ src: t, tokens: e });
              },
            },
            {
              key: "inlineTokens",
              value: function (t) {
                var e,
                  n,
                  r,
                  i,
                  u,
                  a,
                  o = this,
                  s =
                    arguments.length > 1 && void 0 !== arguments[1]
                      ? arguments[1]
                      : [],
                  l = t;
                if (this.tokens.links) {
                  var c = Object.keys(this.tokens.links);
                  if (c.length > 0)
                    for (
                      ;
                      null !=
                      (i = this.tokenizer.rules.inline.reflinkSearch.exec(l));

                    )
                      c.includes(i[0].slice(i[0].lastIndexOf("[") + 1, -1)) &&
                        (l =
                          l.slice(0, i.index) +
                          "[" +
                          ts("a", i[0].length - 2) +
                          "]" +
                          l.slice(
                            this.tokenizer.rules.inline.reflinkSearch.lastIndex,
                          ));
                }
                for (
                  ;
                  null != (i = this.tokenizer.rules.inline.blockSkip.exec(l));

                )
                  l =
                    l.slice(0, i.index) +
                    "[" +
                    ts("a", i[0].length - 2) +
                    "]" +
                    l.slice(this.tokenizer.rules.inline.blockSkip.lastIndex);
                for (
                  ;
                  null != (i = this.tokenizer.rules.inline.escapedEmSt.exec(l));

                )
                  l =
                    l.slice(0, i.index) +
                    "++" +
                    l.slice(this.tokenizer.rules.inline.escapedEmSt.lastIndex);
                for (; t; )
                  if (
                    (u || (a = ""),
                    (u = !1),
                    !(
                      this.options.extensions &&
                      this.options.extensions.inline &&
                      this.options.extensions.inline.some(function (n) {
                        return (
                          !!(e = n.call({ lexer: o }, t, s)) &&
                          ((t = t.substring(e.raw.length)), s.push(e), !0)
                        );
                      })
                    ))
                  )
                    if ((e = this.tokenizer.escape(t)))
                      (t = t.substring(e.raw.length)), s.push(e);
                    else if ((e = this.tokenizer.tag(t)))
                      (t = t.substring(e.raw.length)),
                        (n = s[s.length - 1]) &&
                        "text" === e.type &&
                        "text" === n.type
                          ? ((n.raw += e.raw), (n.text += e.text))
                          : s.push(e);
                    else if ((e = this.tokenizer.link(t)))
                      (t = t.substring(e.raw.length)), s.push(e);
                    else if ((e = this.tokenizer.reflink(t, this.tokens.links)))
                      (t = t.substring(e.raw.length)),
                        (n = s[s.length - 1]) &&
                        "text" === e.type &&
                        "text" === n.type
                          ? ((n.raw += e.raw), (n.text += e.text))
                          : s.push(e);
                    else if ((e = this.tokenizer.emStrong(t, l, a)))
                      (t = t.substring(e.raw.length)), s.push(e);
                    else if ((e = this.tokenizer.codespan(t)))
                      (t = t.substring(e.raw.length)), s.push(e);
                    else if ((e = this.tokenizer.br(t)))
                      (t = t.substring(e.raw.length)), s.push(e);
                    else if ((e = this.tokenizer.del(t)))
                      (t = t.substring(e.raw.length)), s.push(e);
                    else if ((e = this.tokenizer.autolink(t, as)))
                      (t = t.substring(e.raw.length)), s.push(e);
                    else if (
                      this.state.inLink ||
                      !(e = this.tokenizer.url(t, as))
                    ) {
                      if (
                        ((r = t),
                        this.options.extensions &&
                          this.options.extensions.startInline &&
                          (function () {
                            var e = 1 / 0,
                              n = t.slice(1),
                              i = void 0;
                            o.options.extensions.startInline.forEach(
                              function (t) {
                                "number" ==
                                  typeof (i = t.call({ lexer: this }, n)) &&
                                  i >= 0 &&
                                  (e = Math.min(e, i));
                              },
                            ),
                              e < 1 / 0 &&
                                e >= 0 &&
                                (r = t.substring(0, e + 1));
                          })(),
                        (e = this.tokenizer.inlineText(r, us)))
                      )
                        (t = t.substring(e.raw.length)),
                          "_" !== e.raw.slice(-1) && (a = e.raw.slice(-1)),
                          (u = !0),
                          (n = s[s.length - 1]) && "text" === n.type
                            ? ((n.raw += e.raw), (n.text += e.text))
                            : s.push(e);
                      else if (t) {
                        var p = "Infinite loop on byte: " + t.charCodeAt(0);
                        if (this.options.silent) {
                          console.error(p);
                          break;
                        }
                        throw new Error(p);
                      }
                    } else (t = t.substring(e.raw.length)), s.push(e);
                return s;
              },
            },
          ],
          [
            {
              key: "rules",
              get: function () {
                return { block: rs, inline: is };
              },
            },
            {
              key: "lex",
              value: function (e, n) {
                return new t(n).lex(e);
              },
            },
            {
              key: "lexInline",
              value: function (e, n) {
                return new t(n).inlineTokens(e);
              },
            },
          ],
        ),
        t
      );
    })(),
    ss = (function () {
      function t(e) {
        or(this, t), (this.options = e || To);
      }
      return (
        lr(t, [
          {
            key: "code",
            value: function (t, e, n) {
              var r = (e || "").match(/\S*/)[0];
              if (this.options.highlight) {
                var i = this.options.highlight(t, r);
                null != i && i !== t && ((n = !0), (t = i));
              }
              return (
                (t = t.replace(/\n$/, "") + "\n"),
                r
                  ? '<pre><code class="' +
                    this.options.langPrefix +
                    $o(r, !0) +
                    '">' +
                    (n ? t : $o(t, !0)) +
                    "</code></pre>\n"
                  : "<pre><code>" + (n ? t : $o(t, !0)) + "</code></pre>\n"
              );
            },
          },
          {
            key: "blockquote",
            value: function (t) {
              return "<blockquote>\n" + t + "</blockquote>\n";
            },
          },
          {
            key: "html",
            value: function (t) {
              return t;
            },
          },
          {
            key: "heading",
            value: function (t, e, n, r) {
              return this.options.headerIds
                ? "<h" +
                    e +
                    ' id="' +
                    this.options.headerPrefix +
                    r.slug(n) +
                    '">' +
                    t +
                    "</h" +
                    e +
                    ">\n"
                : "<h" + e + ">" + t + "</h" + e + ">\n";
            },
          },
          {
            key: "hr",
            value: function () {
              return this.options.xhtml ? "<hr/>\n" : "<hr>\n";
            },
          },
          {
            key: "list",
            value: function (t, e, n) {
              var r = e ? "ol" : "ul";
              return (
                "<" +
                r +
                (e && 1 !== n ? ' start="' + n + '"' : "") +
                ">\n" +
                t +
                "</" +
                r +
                ">\n"
              );
            },
          },
          {
            key: "listitem",
            value: function (t) {
              return "<li>" + t + "</li>\n";
            },
          },
          {
            key: "checkbox",
            value: function (t) {
              return (
                "<input " +
                (t ? 'checked="" ' : "") +
                'disabled="" type="checkbox"' +
                (this.options.xhtml ? " /" : "") +
                "> "
              );
            },
          },
          {
            key: "paragraph",
            value: function (t) {
              return "<p>" + t + "</p>\n";
            },
          },
          {
            key: "table",
            value: function (t, e) {
              return (
                e && (e = "<tbody>" + e + "</tbody>"),
                "<table>\n<thead>\n" + t + "</thead>\n" + e + "</table>\n"
              );
            },
          },
          {
            key: "tablerow",
            value: function (t) {
              return "<tr>\n" + t + "</tr>\n";
            },
          },
          {
            key: "tablecell",
            value: function (t, e) {
              var n = e.header ? "th" : "td";
              return (
                (e.align
                  ? "<" + n + ' align="' + e.align + '">'
                  : "<" + n + ">") +
                t +
                "</" +
                n +
                ">\n"
              );
            },
          },
          {
            key: "strong",
            value: function (t) {
              return "<strong>" + t + "</strong>";
            },
          },
          {
            key: "em",
            value: function (t) {
              return "<em>" + t + "</em>";
            },
          },
          {
            key: "codespan",
            value: function (t) {
              return "<code>" + t + "</code>";
            },
          },
          {
            key: "br",
            value: function () {
              return this.options.xhtml ? "<br/>" : "<br>";
            },
          },
          {
            key: "del",
            value: function (t) {
              return "<del>" + t + "</del>";
            },
          },
          {
            key: "link",
            value: function (t, e, n) {
              if (
                null ===
                (t = Zo(this.options.sanitize, this.options.baseUrl, t))
              )
                return n;
              var r = '<a href="' + $o(t) + '"';
              return e && (r += ' title="' + e + '"'), (r += ">" + n + "</a>");
            },
          },
          {
            key: "image",
            value: function (t, e, n) {
              if (
                null ===
                (t = Zo(this.options.sanitize, this.options.baseUrl, t))
              )
                return n;
              var r = '<img src="' + t + '" alt="' + n + '"';
              return (
                e && (r += ' title="' + e + '"'),
                (r += this.options.xhtml ? "/>" : ">")
              );
            },
          },
          {
            key: "text",
            value: function (t) {
              return t;
            },
          },
        ]),
        t
      );
    })(),
    ls = (function () {
      function t() {
        or(this, t);
      }
      return (
        lr(t, [
          {
            key: "strong",
            value: function (t) {
              return t;
            },
          },
          {
            key: "em",
            value: function (t) {
              return t;
            },
          },
          {
            key: "codespan",
            value: function (t) {
              return t;
            },
          },
          {
            key: "del",
            value: function (t) {
              return t;
            },
          },
          {
            key: "html",
            value: function (t) {
              return t;
            },
          },
          {
            key: "text",
            value: function (t) {
              return t;
            },
          },
          {
            key: "link",
            value: function (t, e, n) {
              return "" + n;
            },
          },
          {
            key: "image",
            value: function (t, e, n) {
              return "" + n;
            },
          },
          {
            key: "br",
            value: function () {
              return "";
            },
          },
        ]),
        t
      );
    })(),
    cs = (function () {
      function t() {
        or(this, t), (this.seen = {});
      }
      return (
        lr(t, [
          {
            key: "serialize",
            value: function (t) {
              return t
                .toLowerCase()
                .trim()
                .replace(/<[!\/a-z].*?>/gi, "")
                .replace(
                  /[\u2000-\u206F\u2E00-\u2E7F\\'!"#$%&()*+,./:;<=>?@[\]^`{|}~]/g,
                  "",
                )
                .replace(/\s/g, "-");
            },
          },
          {
            key: "getNextSafeSlug",
            value: function (t, e) {
              var n = t,
                r = 0;
              if (this.seen.hasOwnProperty(n)) {
                r = this.seen[t];
                do {
                  n = t + "-" + ++r;
                } while (this.seen.hasOwnProperty(n));
              }
              return e || ((this.seen[t] = r), (this.seen[n] = 0)), n;
            },
          },
          {
            key: "slug",
            value: function (t) {
              var e =
                  arguments.length > 1 && void 0 !== arguments[1]
                    ? arguments[1]
                    : {},
                n = this.serialize(t);
              return this.getNextSafeSlug(n, e.dryrun);
            },
          },
        ]),
        t
      );
    })(),
    ps = (function () {
      function t(e) {
        or(this, t),
          (this.options = e || To),
          (this.options.renderer = this.options.renderer || new ss()),
          (this.renderer = this.options.renderer),
          (this.renderer.options = this.options),
          (this.textRenderer = new ls()),
          (this.slugger = new cs());
      }
      return (
        lr(
          t,
          [
            {
              key: "parse",
              value: function (t) {
                var e,
                  n,
                  r,
                  i,
                  u,
                  a,
                  o,
                  s,
                  l,
                  c,
                  p,
                  d,
                  f,
                  h,
                  g,
                  D,
                  m,
                  v,
                  y,
                  k =
                    !(arguments.length > 1 && void 0 !== arguments[1]) ||
                    arguments[1],
                  E = "",
                  x = t.length;
                for (e = 0; e < x; e++)
                  if (
                    ((c = t[e]),
                    !(
                      this.options.extensions &&
                      this.options.extensions.renderers &&
                      this.options.extensions.renderers[c.type]
                    ) ||
                      (!1 ===
                        (y = this.options.extensions.renderers[c.type].call(
                          { parser: this },
                          c,
                        )) &&
                        [
                          "space",
                          "hr",
                          "heading",
                          "code",
                          "table",
                          "blockquote",
                          "list",
                          "html",
                          "paragraph",
                          "text",
                        ].includes(c.type)))
                  )
                    switch (c.type) {
                      case "space":
                        continue;
                      case "hr":
                        E += this.renderer.hr();
                        continue;
                      case "heading":
                        E += this.renderer.heading(
                          this.parseInline(c.tokens),
                          c.depth,
                          Mo(this.parseInline(c.tokens, this.textRenderer)),
                          this.slugger,
                        );
                        continue;
                      case "code":
                        E += this.renderer.code(c.text, c.lang, c.escaped);
                        continue;
                      case "table":
                        for (
                          s = "", o = "", i = c.header.length, n = 0;
                          n < i;
                          n++
                        )
                          o += this.renderer.tablecell(
                            this.parseInline(c.header[n].tokens),
                            { header: !0, align: c.align[n] },
                          );
                        for (
                          s += this.renderer.tablerow(o),
                            l = "",
                            i = c.rows.length,
                            n = 0;
                          n < i;
                          n++
                        ) {
                          for (
                            o = "", u = (a = c.rows[n]).length, r = 0;
                            r < u;
                            r++
                          )
                            o += this.renderer.tablecell(
                              this.parseInline(a[r].tokens),
                              { header: !1, align: c.align[r] },
                            );
                          l += this.renderer.tablerow(o);
                        }
                        E += this.renderer.table(s, l);
                        continue;
                      case "blockquote":
                        (l = this.parse(c.tokens)),
                          (E += this.renderer.blockquote(l));
                        continue;
                      case "list":
                        for (
                          p = c.ordered,
                            d = c.start,
                            f = c.loose,
                            i = c.items.length,
                            l = "",
                            n = 0;
                          n < i;
                          n++
                        )
                          (D = (g = c.items[n]).checked),
                            (m = g.task),
                            (h = ""),
                            g.task &&
                              ((v = this.renderer.checkbox(D)),
                              f
                                ? g.tokens.length > 0 &&
                                  "paragraph" === g.tokens[0].type
                                  ? ((g.tokens[0].text =
                                      v + " " + g.tokens[0].text),
                                    g.tokens[0].tokens &&
                                      g.tokens[0].tokens.length > 0 &&
                                      "text" === g.tokens[0].tokens[0].type &&
                                      (g.tokens[0].tokens[0].text =
                                        v + " " + g.tokens[0].tokens[0].text))
                                  : g.tokens.unshift({ type: "text", text: v })
                                : (h += v)),
                            (h += this.parse(g.tokens, f)),
                            (l += this.renderer.listitem(h, m, D));
                        E += this.renderer.list(l, p, d);
                        continue;
                      case "html":
                        E += this.renderer.html(c.text);
                        continue;
                      case "paragraph":
                        E += this.renderer.paragraph(
                          this.parseInline(c.tokens),
                        );
                        continue;
                      case "text":
                        for (
                          l = c.tokens ? this.parseInline(c.tokens) : c.text;
                          e + 1 < x && "text" === t[e + 1].type;

                        )
                          l +=
                            "\n" +
                            ((c = t[++e]).tokens
                              ? this.parseInline(c.tokens)
                              : c.text);
                        E += k ? this.renderer.paragraph(l) : l;
                        continue;
                      default:
                        var A =
                          'Token with "' + c.type + '" type was not found.';
                        if (this.options.silent) return void console.error(A);
                        throw new Error(A);
                    }
                  else E += y || "";
                return E;
              },
            },
            {
              key: "parseInline",
              value: function (t, e) {
                e = e || this.renderer;
                var n,
                  r,
                  i,
                  u = "",
                  a = t.length;
                for (n = 0; n < a; n++)
                  if (
                    ((r = t[n]),
                    !(
                      this.options.extensions &&
                      this.options.extensions.renderers &&
                      this.options.extensions.renderers[r.type]
                    ) ||
                      (!1 ===
                        (i = this.options.extensions.renderers[r.type].call(
                          { parser: this },
                          r,
                        )) &&
                        [
                          "escape",
                          "html",
                          "link",
                          "image",
                          "strong",
                          "em",
                          "codespan",
                          "br",
                          "del",
                          "text",
                        ].includes(r.type)))
                  )
                    switch (r.type) {
                      case "escape":
                        u += e.text(r.text);
                        break;
                      case "html":
                        u += e.html(r.text);
                        break;
                      case "link":
                        u += e.link(
                          r.href,
                          r.title,
                          this.parseInline(r.tokens, e),
                        );
                        break;
                      case "image":
                        u += e.image(r.href, r.title, r.text);
                        break;
                      case "strong":
                        u += e.strong(this.parseInline(r.tokens, e));
                        break;
                      case "em":
                        u += e.em(this.parseInline(r.tokens, e));
                        break;
                      case "codespan":
                        u += e.codespan(r.text);
                        break;
                      case "br":
                        u += e.br();
                        break;
                      case "del":
                        u += e.del(this.parseInline(r.tokens, e));
                        break;
                      case "text":
                        u += e.text(r.text);
                        break;
                      default:
                        var o =
                          'Token with "' + r.type + '" type was not found.';
                        if (this.options.silent) return void console.error(o);
                        throw new Error(o);
                    }
                  else u += i || "";
                return u;
              },
            },
          ],
          [
            {
              key: "parse",
              value: function (e, n) {
                return new t(n).parse(e);
              },
            },
            {
              key: "parseInline",
              value: function (e, n) {
                return new t(n).parseInline(e);
              },
            },
          ],
        ),
        t
      );
    })();
  function ds(t, e, n) {
    if (null == t)
      throw new Error("marked(): input parameter is undefined or null");
    if ("string" != typeof t)
      throw new Error(
        "marked(): input parameter is of type " +
          Object.prototype.toString.call(t) +
          ", string expected",
      );
    if (
      ("function" == typeof e && ((n = e), (e = null)),
      Xo((e = Qo({}, ds.defaults, e || {}))),
      n)
    ) {
      var r,
        i = e.highlight;
      try {
        r = os.lex(t, e);
      } catch (t) {
        return n(t);
      }
      var u = function (t) {
        var u;
        if (!t)
          try {
            e.walkTokens && ds.walkTokens(r, e.walkTokens),
              (u = ps.parse(r, e));
          } catch (e) {
            t = e;
          }
        return (e.highlight = i), t ? n(t) : n(null, u);
      };
      if (!i || i.length < 3) return u();
      if ((delete e.highlight, !r.length)) return u();
      var a = 0;
      return (
        ds.walkTokens(r, function (t) {
          "code" === t.type &&
            (a++,
            setTimeout(function () {
              i(t.text, t.lang, function (e, n) {
                if (e) return u(e);
                null != n && n !== t.text && ((t.text = n), (t.escaped = !0)),
                  0 === --a && u();
              });
            }, 0));
        }),
        void (0 === a && u())
      );
    }
    try {
      var o = os.lex(t, e);
      return e.walkTokens && ds.walkTokens(o, e.walkTokens), ps.parse(o, e);
    } catch (t) {
      if (
        ((t.message +=
          "\nPlease report this to https://github.com/markedjs/marked."),
        e.silent)
      )
        return (
          "<p>An error occurred:</p><pre>" + $o(t.message + "", !0) + "</pre>"
        );
      throw t;
    }
  }
  (ds.options = ds.setOptions =
    function (t) {
      var e;
      return Qo(ds.defaults, t), (e = ds.defaults), (To = e), ds;
    }),
    (ds.getDefaults = Bo),
    (ds.defaults = To),
    (ds.use = function () {
      for (var t = arguments.length, e = new Array(t), n = 0; n < t; n++)
        e[n] = arguments[n];
      var r,
        i = Qo.apply(void 0, [{}].concat(e)),
        u = ds.defaults.extensions || { renderers: {}, childTokens: {} };
      e.forEach(function (t) {
        if (
          (t.extensions &&
            ((r = !0),
            t.extensions.forEach(function (t) {
              if (!t.name) throw new Error("extension name required");
              if (t.renderer) {
                var e = u.renderers ? u.renderers[t.name] : null;
                u.renderers[t.name] = e
                  ? function () {
                      for (
                        var n = arguments.length, r = new Array(n), i = 0;
                        i < n;
                        i++
                      )
                        r[i] = arguments[i];
                      var u = t.renderer.apply(this, r);
                      return !1 === u && (u = e.apply(this, r)), u;
                    }
                  : t.renderer;
              }
              if (t.tokenizer) {
                if (!t.level || ("block" !== t.level && "inline" !== t.level))
                  throw new Error(
                    "extension level must be 'block' or 'inline'",
                  );
                u[t.level]
                  ? u[t.level].unshift(t.tokenizer)
                  : (u[t.level] = [t.tokenizer]),
                  t.start &&
                    ("block" === t.level
                      ? u.startBlock
                        ? u.startBlock.push(t.start)
                        : (u.startBlock = [t.start])
                      : "inline" === t.level &&
                        (u.startInline
                          ? u.startInline.push(t.start)
                          : (u.startInline = [t.start])));
              }
              t.childTokens && (u.childTokens[t.name] = t.childTokens);
            })),
          t.renderer &&
            (function () {
              var e = ds.defaults.renderer || new ss(),
                n = function (n) {
                  var r = e[n];
                  e[n] = function () {
                    for (
                      var i = arguments.length, u = new Array(i), a = 0;
                      a < i;
                      a++
                    )
                      u[a] = arguments[a];
                    var o = t.renderer[n].apply(e, u);
                    return !1 === o && (o = r.apply(e, u)), o;
                  };
                };
              for (var r in t.renderer) n(r);
              i.renderer = e;
            })(),
          t.tokenizer &&
            (function () {
              var e = ds.defaults.tokenizer || new ns(),
                n = function (n) {
                  var r = e[n];
                  e[n] = function () {
                    for (
                      var i = arguments.length, u = new Array(i), a = 0;
                      a < i;
                      a++
                    )
                      u[a] = arguments[a];
                    var o = t.tokenizer[n].apply(e, u);
                    return !1 === o && (o = r.apply(e, u)), o;
                  };
                };
              for (var r in t.tokenizer) n(r);
              i.tokenizer = e;
            })(),
          t.walkTokens)
        ) {
          var e = ds.defaults.walkTokens;
          i.walkTokens = function (n) {
            t.walkTokens.call(this, n), e && e.call(this, n);
          };
        }
        r && (i.extensions = u), ds.setOptions(i);
      });
    }),
    (ds.walkTokens = function (t, e) {
      var n,
        r = fr(t);
      try {
        var i = function () {
          var t = n.value;
          switch ((e.call(ds, t), t.type)) {
            case "table":
              var r,
                i = fr(t.header);
              try {
                for (i.s(); !(r = i.n()).done; ) {
                  var u = r.value;
                  ds.walkTokens(u.tokens, e);
                }
              } catch (t) {
                i.e(t);
              } finally {
                i.f();
              }
              var a,
                o = fr(t.rows);
              try {
                for (o.s(); !(a = o.n()).done; ) {
                  var s,
                    l = fr(a.value);
                  try {
                    for (l.s(); !(s = l.n()).done; ) {
                      var c = s.value;
                      ds.walkTokens(c.tokens, e);
                    }
                  } catch (t) {
                    l.e(t);
                  } finally {
                    l.f();
                  }
                }
              } catch (t) {
                o.e(t);
              } finally {
                o.f();
              }
              break;
            case "list":
              ds.walkTokens(t.items, e);
              break;
            default:
              ds.defaults.extensions &&
              ds.defaults.extensions.childTokens &&
              ds.defaults.extensions.childTokens[t.type]
                ? ds.defaults.extensions.childTokens[t.type].forEach(
                    function (n) {
                      ds.walkTokens(t[n], e);
                    },
                  )
                : t.tokens && ds.walkTokens(t.tokens, e);
          }
        };
        for (r.s(); !(n = r.n()).done; ) i();
      } catch (t) {
        r.e(t);
      } finally {
        r.f();
      }
    }),
    (ds.parseInline = function (t, e) {
      if (null == t)
        throw new Error(
          "marked.parseInline(): input parameter is undefined or null",
        );
      if ("string" != typeof t)
        throw new Error(
          "marked.parseInline(): input parameter is of type " +
            Object.prototype.toString.call(t) +
            ", string expected",
        );
      Xo((e = Qo({}, ds.defaults, e || {})));
      try {
        var n = os.lexInline(t, e);
        return (
          e.walkTokens && ds.walkTokens(n, e.walkTokens), ps.parseInline(n, e)
        );
      } catch (t) {
        if (
          ((t.message +=
            "\nPlease report this to https://github.com/markedjs/marked."),
          e.silent)
        )
          return (
            "<p>An error occurred:</p><pre>" + $o(t.message + "", !0) + "</pre>"
          );
        throw t;
      }
    }),
    (ds.Parser = ps),
    (ds.parser = ps.parse),
    (ds.Renderer = ss),
    (ds.TextRenderer = ls),
    (ds.Lexer = os),
    (ds.lexer = os.lex),
    (ds.Tokenizer = ns),
    (ds.Slugger = cs),
    (ds.parse = ds);
  return function () {
    var t,
      e,
      n = null;
    function r() {
      if (n && !n.closed) n.focus();
      else {
        if (
          (((n = window.open(
            "about:blank",
            "reveal.js - Notes",
            "width=1100,height=700",
          )).marked = ds),
          n.document.write(
            "\x3c!--\n\tNOTE: You need to build the notes plugin after making changes to this file.\n--\x3e\n<html lang=\"en\">\n\t<head>\n\t\t<meta charset=\"utf-8\">\n\n\t\t<title>reveal.js - Speaker View</title>\n\n\t\t<style>\n\t\t\tbody {\n\t\t\t\tfont-family: Helvetica;\n\t\t\t\tfont-size: 18px;\n\t\t\t}\n\n\t\t\t#current-slide,\n\t\t\t#upcoming-slide,\n\t\t\t#speaker-controls {\n\t\t\t\tpadding: 6px;\n\t\t\t\tbox-sizing: border-box;\n\t\t\t\t-moz-box-sizing: border-box;\n\t\t\t}\n\n\t\t\t#current-slide iframe,\n\t\t\t#upcoming-slide iframe {\n\t\t\t\twidth: 100%;\n\t\t\t\theight: 100%;\n\t\t\t\tborder: 1px solid #ddd;\n\t\t\t}\n\n\t\t\t#current-slide .label,\n\t\t\t#upcoming-slide .label {\n\t\t\t\tposition: absolute;\n\t\t\t\ttop: 10px;\n\t\t\t\tleft: 10px;\n\t\t\t\tz-index: 2;\n\t\t\t}\n\n\t\t\t#connection-status {\n\t\t\t\tposition: absolute;\n\t\t\t\ttop: 0;\n\t\t\t\tleft: 0;\n\t\t\t\twidth: 100%;\n\t\t\t\theight: 100%;\n\t\t\t\tz-index: 20;\n\t\t\t\tpadding: 30% 20% 20% 20%;\n\t\t\t\tfont-size: 18px;\n\t\t\t\tcolor: #222;\n\t\t\t\tbackground: #fff;\n\t\t\t\ttext-align: center;\n\t\t\t\tbox-sizing: border-box;\n\t\t\t\tline-height: 1.4;\n\t\t\t}\n\n\t\t\t.overlay-element {\n\t\t\t\theight: 34px;\n\t\t\t\tline-height: 34px;\n\t\t\t\tpadding: 0 10px;\n\t\t\t\ttext-shadow: none;\n\t\t\t\tbackground: rgba( 220, 220, 220, 0.8 );\n\t\t\t\tcolor: #222;\n\t\t\t\tfont-size: 14px;\n\t\t\t}\n\n\t\t\t.overlay-element.interactive:hover {\n\t\t\t\tbackground: rgba( 220, 220, 220, 1 );\n\t\t\t}\n\n\t\t\t#current-slide {\n\t\t\t\tposition: absolute;\n\t\t\t\twidth: 60%;\n\t\t\t\theight: 100%;\n\t\t\t\ttop: 0;\n\t\t\t\tleft: 0;\n\t\t\t\tpadding-right: 0;\n\t\t\t}\n\n\t\t\t#upcoming-slide {\n\t\t\t\tposition: absolute;\n\t\t\t\twidth: 40%;\n\t\t\t\theight: 40%;\n\t\t\t\tright: 0;\n\t\t\t\ttop: 0;\n\t\t\t}\n\n\t\t\t/* Speaker controls */\n\t\t\t#speaker-controls {\n\t\t\t\tposition: absolute;\n\t\t\t\ttop: 40%;\n\t\t\t\tright: 0;\n\t\t\t\twidth: 40%;\n\t\t\t\theight: 60%;\n\t\t\t\toverflow: auto;\n\t\t\t\tfont-size: 18px;\n\t\t\t}\n\n\t\t\t\t.speaker-controls-time.hidden,\n\t\t\t\t.speaker-controls-notes.hidden {\n\t\t\t\t\tdisplay: none;\n\t\t\t\t}\n\n\t\t\t\t.speaker-controls-time .label,\n\t\t\t\t.speaker-controls-pace .label,\n\t\t\t\t.speaker-controls-notes .label {\n\t\t\t\t\ttext-transform: uppercase;\n\t\t\t\t\tfont-weight: normal;\n\t\t\t\t\tfont-size: 0.66em;\n\t\t\t\t\tcolor: #666;\n\t\t\t\t\tmargin: 0;\n\t\t\t\t}\n\n\t\t\t\t.speaker-controls-time, .speaker-controls-pace {\n\t\t\t\t\tborder-bottom: 1px solid rgba( 200, 200, 200, 0.5 );\n\t\t\t\t\tmargin-bottom: 10px;\n\t\t\t\t\tpadding: 10px 16px;\n\t\t\t\t\tpadding-bottom: 20px;\n\t\t\t\t\tcursor: pointer;\n\t\t\t\t}\n\n\t\t\t\t.speaker-controls-time .reset-button {\n\t\t\t\t\topacity: 0;\n\t\t\t\t\tfloat: right;\n\t\t\t\t\tcolor: #666;\n\t\t\t\t\ttext-decoration: none;\n\t\t\t\t}\n\t\t\t\t.speaker-controls-time:hover .reset-button {\n\t\t\t\t\topacity: 1;\n\t\t\t\t}\n\n\t\t\t\t.speaker-controls-time .timer,\n\t\t\t\t.speaker-controls-time .clock {\n\t\t\t\t\twidth: 50%;\n\t\t\t\t}\n\n\t\t\t\t.speaker-controls-time .timer,\n\t\t\t\t.speaker-controls-time .clock,\n\t\t\t\t.speaker-controls-time .pacing .hours-value,\n\t\t\t\t.speaker-controls-time .pacing .minutes-value,\n\t\t\t\t.speaker-controls-time .pacing .seconds-value {\n\t\t\t\t\tfont-size: 1.9em;\n\t\t\t\t}\n\n\t\t\t\t.speaker-controls-time .timer {\n\t\t\t\t\tfloat: left;\n\t\t\t\t}\n\n\t\t\t\t.speaker-controls-time .clock {\n\t\t\t\t\tfloat: right;\n\t\t\t\t\ttext-align: right;\n\t\t\t\t}\n\n\t\t\t\t.speaker-controls-time span.mute {\n\t\t\t\t\topacity: 0.3;\n\t\t\t\t}\n\n\t\t\t\t.speaker-controls-time .pacing-title {\n\t\t\t\t\tmargin-top: 5px;\n\t\t\t\t}\n\n\t\t\t\t.speaker-controls-time .pacing.ahead {\n\t\t\t\t\tcolor: blue;\n\t\t\t\t}\n\n\t\t\t\t.speaker-controls-time .pacing.on-track {\n\t\t\t\t\tcolor: green;\n\t\t\t\t}\n\n\t\t\t\t.speaker-controls-time .pacing.behind {\n\t\t\t\t\tcolor: red;\n\t\t\t\t}\n\n\t\t\t\t.speaker-controls-notes {\n\t\t\t\t\tpadding: 10px 16px;\n\t\t\t\t}\n\n\t\t\t\t.speaker-controls-notes .value {\n\t\t\t\t\tmargin-top: 5px;\n\t\t\t\t\tline-height: 1.4;\n\t\t\t\t\tfont-size: 1.2em;\n\t\t\t\t}\n\n\t\t\t/* Layout selector */\n\t\t\t#speaker-layout {\n\t\t\t\tposition: absolute;\n\t\t\t\ttop: 10px;\n\t\t\t\tright: 10px;\n\t\t\t\tcolor: #222;\n\t\t\t\tz-index: 10;\n\t\t\t}\n\t\t\t\t#speaker-layout select {\n\t\t\t\t\tposition: absolute;\n\t\t\t\t\twidth: 100%;\n\t\t\t\t\theight: 100%;\n\t\t\t\t\ttop: 0;\n\t\t\t\t\tleft: 0;\n\t\t\t\t\tborder: 0;\n\t\t\t\t\tbox-shadow: 0;\n\t\t\t\t\tcursor: pointer;\n\t\t\t\t\topacity: 0;\n\n\t\t\t\t\tfont-size: 1em;\n\t\t\t\t\tbackground-color: transparent;\n\n\t\t\t\t\t-moz-appearance: none;\n\t\t\t\t\t-webkit-appearance: none;\n\t\t\t\t\t-webkit-tap-highlight-color: rgba(0, 0, 0, 0);\n\t\t\t\t}\n\n\t\t\t\t#speaker-layout select:focus {\n\t\t\t\t\toutline: none;\n\t\t\t\t\tbox-shadow: none;\n\t\t\t\t}\n\n\t\t\t.clear {\n\t\t\t\tclear: both;\n\t\t\t}\n\n\t\t\t/* Speaker layout: Wide */\n\t\t\tbody[data-speaker-layout=\"wide\"] #current-slide,\n\t\t\tbody[data-speaker-layout=\"wide\"] #upcoming-slide {\n\t\t\t\twidth: 50%;\n\t\t\t\theight: 45%;\n\t\t\t\tpadding: 6px;\n\t\t\t}\n\n\t\t\tbody[data-speaker-layout=\"wide\"] #current-slide {\n\t\t\t\ttop: 0;\n\t\t\t\tleft: 0;\n\t\t\t}\n\n\t\t\tbody[data-speaker-layout=\"wide\"] #upcoming-slide {\n\t\t\t\ttop: 0;\n\t\t\t\tleft: 50%;\n\t\t\t}\n\n\t\t\tbody[data-speaker-layout=\"wide\"] #speaker-controls {\n\t\t\t\ttop: 45%;\n\t\t\t\tleft: 0;\n\t\t\t\twidth: 100%;\n\t\t\t\theight: 50%;\n\t\t\t\tfont-size: 1.25em;\n\t\t\t}\n\n\t\t\t/* Speaker layout: Tall */\n\t\t\tbody[data-speaker-layout=\"tall\"] #current-slide,\n\t\t\tbody[data-speaker-layout=\"tall\"] #upcoming-slide {\n\t\t\t\twidth: 45%;\n\t\t\t\theight: 50%;\n\t\t\t\tpadding: 6px;\n\t\t\t}\n\n\t\t\tbody[data-speaker-layout=\"tall\"] #current-slide {\n\t\t\t\ttop: 0;\n\t\t\t\tleft: 0;\n\t\t\t}\n\n\t\t\tbody[data-speaker-layout=\"tall\"] #upcoming-slide {\n\t\t\t\ttop: 50%;\n\t\t\t\tleft: 0;\n\t\t\t}\n\n\t\t\tbody[data-speaker-layout=\"tall\"] #speaker-controls {\n\t\t\t\tpadding-top: 40px;\n\t\t\t\ttop: 0;\n\t\t\t\tleft: 45%;\n\t\t\t\twidth: 55%;\n\t\t\t\theight: 100%;\n\t\t\t\tfont-size: 1.25em;\n\t\t\t}\n\n\t\t\t/* Speaker layout: Notes only */\n\t\t\tbody[data-speaker-layout=\"notes-only\"] #current-slide,\n\t\t\tbody[data-speaker-layout=\"notes-only\"] #upcoming-slide {\n\t\t\t\tdisplay: none;\n\t\t\t}\n\n\t\t\tbody[data-speaker-layout=\"notes-only\"] #speaker-controls {\n\t\t\t\tpadding-top: 40px;\n\t\t\t\ttop: 0;\n\t\t\t\tleft: 0;\n\t\t\t\twidth: 100%;\n\t\t\t\theight: 100%;\n\t\t\t\tfont-size: 1.25em;\n\t\t\t}\n\n\t\t\t@media screen and (max-width: 1080px) {\n\t\t\t\tbody[data-speaker-layout=\"default\"] #speaker-controls {\n\t\t\t\t\tfont-size: 16px;\n\t\t\t\t}\n\t\t\t}\n\n\t\t\t@media screen and (max-width: 900px) {\n\t\t\t\tbody[data-speaker-layout=\"default\"] #speaker-controls {\n\t\t\t\t\tfont-size: 14px;\n\t\t\t\t}\n\t\t\t}\n\n\t\t\t@media screen and (max-width: 800px) {\n\t\t\t\tbody[data-speaker-layout=\"default\"] #speaker-controls {\n\t\t\t\t\tfont-size: 12px;\n\t\t\t\t}\n\t\t\t}\n\n\t\t</style>\n\t</head>\n\n\t<body>\n\n\t\t<div id=\"connection-status\">Loading speaker view...</div>\n\n\t\t<div id=\"current-slide\"></div>\n\t\t<div id=\"upcoming-slide\"><span class=\"overlay-element label\">Upcoming</span></div>\n\t\t<div id=\"speaker-controls\">\n\t\t\t<div class=\"speaker-controls-time\">\n\t\t\t\t<h4 class=\"label\">Time <span class=\"reset-button\">Click to Reset</span></h4>\n\t\t\t\t<div class=\"clock\">\n\t\t\t\t\t<span class=\"clock-value\">0:00 AM</span>\n\t\t\t\t</div>\n\t\t\t\t<div class=\"timer\">\n\t\t\t\t\t<span class=\"hours-value\">00</span><span class=\"minutes-value\">:00</span><span class=\"seconds-value\">:00</span>\n\t\t\t\t</div>\n\t\t\t\t<div class=\"clear\"></div>\n\n\t\t\t\t<h4 class=\"label pacing-title\" style=\"display: none\">Pacing – Time to finish current slide</h4>\n\t\t\t\t<div class=\"pacing\" style=\"display: none\">\n\t\t\t\t\t<span class=\"hours-value\">00</span><span class=\"minutes-value\">:00</span><span class=\"seconds-value\">:00</span>\n\t\t\t\t</div>\n\t\t\t</div>\n\n\t\t\t<div class=\"speaker-controls-notes hidden\">\n\t\t\t\t<h4 class=\"label\">Notes</h4>\n\t\t\t\t<div class=\"value\"></div>\n\t\t\t</div>\n\t\t</div>\n\t\t<div id=\"speaker-layout\" class=\"overlay-element interactive\">\n\t\t\t<span class=\"speaker-layout-label\"></span>\n\t\t\t<select class=\"speaker-layout-dropdown\"></select>\n\t\t</div>\n\n\t\t<script>\n\n\t\t\t(function() {\n\n\t\t\t\tvar notes,\n\t\t\t\t\tnotesValue,\n\t\t\t\t\tcurrentState,\n\t\t\t\t\tcurrentSlide,\n\t\t\t\t\tupcomingSlide,\n\t\t\t\t\tlayoutLabel,\n\t\t\t\t\tlayoutDropdown,\n\t\t\t\t\tpendingCalls = {},\n\t\t\t\t\tlastRevealApiCallId = 0,\n\t\t\t\t\tconnected = false,\n\t\t\t\t\twhitelistedWindows = [window.opener];\n\n\t\t\t\tvar SPEAKER_LAYOUTS = {\n\t\t\t\t\t'default': 'Default',\n\t\t\t\t\t'wide': 'Wide',\n\t\t\t\t\t'tall': 'Tall',\n\t\t\t\t\t'notes-only': 'Notes only'\n\t\t\t\t};\n\n\t\t\t\tsetupLayout();\n\n\t\t\t\tvar connectionStatus = document.querySelector( '#connection-status' );\n\t\t\t\tvar connectionTimeout = setTimeout( function() {\n\t\t\t\t\tconnectionStatus.innerHTML = 'Error connecting to main window.<br>Please try closing and reopening the speaker view.';\n\t\t\t\t}, 5000 );\n;\n\t\t\t\twindow.addEventListener( 'message', function( event ) {\n\n\t\t\t\t\t// Validate the origin of this message to prevent XSS\n\t\t\t\t\tif( window.location.origin !== event.origin && whitelistedWindows.indexOf( event.source ) === -1 ) {\n\t\t\t\t\t\treturn;\n\t\t\t\t\t}\n\n\t\t\t\t\tclearTimeout( connectionTimeout );\n\t\t\t\t\tconnectionStatus.style.display = 'none';\n\n\t\t\t\t\tvar data = JSON.parse( event.data );\n\n\t\t\t\t\t// The overview mode is only useful to the reveal.js instance\n\t\t\t\t\t// where navigation occurs so we don't sync it\n\t\t\t\t\tif( data.state ) delete data.state.overview;\n\n\t\t\t\t\t// Messages sent by the notes plugin inside of the main window\n\t\t\t\t\tif( data && data.namespace === 'reveal-notes' ) {\n\t\t\t\t\t\tif( data.type === 'connect' ) {\n\t\t\t\t\t\t\thandleConnectMessage( data );\n\t\t\t\t\t\t}\n\t\t\t\t\t\telse if( data.type === 'state' ) {\n\t\t\t\t\t\t\thandleStateMessage( data );\n\t\t\t\t\t\t}\n\t\t\t\t\t\telse if( data.type === 'return' ) {\n\t\t\t\t\t\t\tpendingCalls[data.callId](data.result);\n\t\t\t\t\t\t\tdelete pendingCalls[data.callId];\n\t\t\t\t\t\t}\n\t\t\t\t\t}\n\t\t\t\t\t// Messages sent by the reveal.js inside of the current slide preview\n\t\t\t\t\telse if( data && data.namespace === 'reveal' ) {\n\t\t\t\t\t\tif( /ready/.test( data.eventName ) ) {\n\t\t\t\t\t\t\t// Send a message back to notify that the handshake is complete\n\t\t\t\t\t\t\twindow.opener.postMessage( JSON.stringify({ namespace: 'reveal-notes', type: 'connected'} ), '*' );\n\t\t\t\t\t\t}\n\t\t\t\t\t\telse if( /slidechanged|fragmentshown|fragmenthidden|paused|resumed/.test( data.eventName ) && currentState !== JSON.stringify( data.state ) ) {\n\n\t\t\t\t\t\t\tdispatchStateToMainWindow( data.state );\n\n\t\t\t\t\t\t}\n\t\t\t\t\t}\n\n\t\t\t\t} );\n\n\t\t\t\t/**\n\t\t\t\t * Updates the presentation in the main window to match the state\n\t\t\t\t * of the presentation in the notes window.\n\t\t\t\t */\n\t\t\t\tconst dispatchStateToMainWindow = debounce(( state ) => {\n\t\t\t\t\twindow.opener.postMessage( JSON.stringify({ method: 'setState', args: [ state ]} ), '*' );\n\t\t\t\t}, 500);\n\n\t\t\t\t/**\n\t\t\t\t * Asynchronously calls the Reveal.js API of the main frame.\n\t\t\t\t */\n\t\t\t\tfunction callRevealApi( methodName, methodArguments, callback ) {\n\n\t\t\t\t\tvar callId = ++lastRevealApiCallId;\n\t\t\t\t\tpendingCalls[callId] = callback;\n\t\t\t\t\twindow.opener.postMessage( JSON.stringify( {\n\t\t\t\t\t\tnamespace: 'reveal-notes',\n\t\t\t\t\t\ttype: 'call',\n\t\t\t\t\t\tcallId: callId,\n\t\t\t\t\t\tmethodName: methodName,\n\t\t\t\t\t\targuments: methodArguments\n\t\t\t\t\t} ), '*' );\n\n\t\t\t\t}\n\n\t\t\t\t/**\n\t\t\t\t * Called when the main window is trying to establish a\n\t\t\t\t * connection.\n\t\t\t\t */\n\t\t\t\tfunction handleConnectMessage( data ) {\n\n\t\t\t\t\tif( connected === false ) {\n\t\t\t\t\t\tconnected = true;\n\n\t\t\t\t\t\tsetupIframes( data );\n\t\t\t\t\t\tsetupKeyboard();\n\t\t\t\t\t\tsetupNotes();\n\t\t\t\t\t\tsetupTimer();\n\t\t\t\t\t\tsetupHeartbeat();\n\t\t\t\t\t}\n\n\t\t\t\t}\n\n\t\t\t\t/**\n\t\t\t\t * Called when the main window sends an updated state.\n\t\t\t\t */\n\t\t\t\tfunction handleStateMessage( data ) {\n\n\t\t\t\t\t// Store the most recently set state to avoid circular loops\n\t\t\t\t\t// applying the same state\n\t\t\t\t\tcurrentState = JSON.stringify( data.state );\n\n\t\t\t\t\t// No need for updating the notes in case of fragment changes\n\t\t\t\t\tif ( data.notes ) {\n\t\t\t\t\t\tnotes.classList.remove( 'hidden' );\n\t\t\t\t\t\tnotesValue.style.whiteSpace = data.whitespace;\n\t\t\t\t\t\tif( data.markdown ) {\n\t\t\t\t\t\t\tnotesValue.innerHTML = marked( data.notes );\n\t\t\t\t\t\t}\n\t\t\t\t\t\telse {\n\t\t\t\t\t\t\tnotesValue.innerHTML = data.notes;\n\t\t\t\t\t\t}\n\t\t\t\t\t}\n\t\t\t\t\telse {\n\t\t\t\t\t\tnotes.classList.add( 'hidden' );\n\t\t\t\t\t}\n\n\t\t\t\t\t// Update the note slides\n\t\t\t\t\tcurrentSlide.contentWindow.postMessage( JSON.stringify({ method: 'setState', args: [ data.state ] }), '*' );\n\t\t\t\t\tupcomingSlide.contentWindow.postMessage( JSON.stringify({ method: 'setState', args: [ data.state ] }), '*' );\n\t\t\t\t\tupcomingSlide.contentWindow.postMessage( JSON.stringify({ method: 'next' }), '*' );\n\n\t\t\t\t}\n\n\t\t\t\t// Limit to max one state update per X ms\n\t\t\t\thandleStateMessage = debounce( handleStateMessage, 200 );\n\n\t\t\t\t/**\n\t\t\t\t * Forward keyboard events to the current slide window.\n\t\t\t\t * This enables keyboard events to work even if focus\n\t\t\t\t * isn't set on the current slide iframe.\n\t\t\t\t *\n\t\t\t\t * Block F5 default handling, it reloads and disconnects\n\t\t\t\t * the speaker notes window.\n\t\t\t\t */\n\t\t\t\tfunction setupKeyboard() {\n\n\t\t\t\t\tdocument.addEventListener( 'keydown', function( event ) {\n\t\t\t\t\t\tif( event.keyCode === 116 || ( event.metaKey && event.keyCode === 82 ) ) {\n\t\t\t\t\t\t\tevent.preventDefault();\n\t\t\t\t\t\t\treturn false;\n\t\t\t\t\t\t}\n\t\t\t\t\t\tcurrentSlide.contentWindow.postMessage( JSON.stringify({ method: 'triggerKey', args: [ event.keyCode ] }), '*' );\n\t\t\t\t\t} );\n\n\t\t\t\t}\n\n\t\t\t\t/**\n\t\t\t\t * Creates the preview iframes.\n\t\t\t\t */\n\t\t\t\tfunction setupIframes( data ) {\n\n\t\t\t\t\tvar params = [\n\t\t\t\t\t\t'receiver',\n\t\t\t\t\t\t'progress=false',\n\t\t\t\t\t\t'history=false',\n\t\t\t\t\t\t'transition=none',\n\t\t\t\t\t\t'autoSlide=0',\n\t\t\t\t\t\t'backgroundTransition=none'\n\t\t\t\t\t].join( '&' );\n\n\t\t\t\t\tvar urlSeparator = /\\?/.test(data.url) ? '&' : '?';\n\t\t\t\t\tvar hash = '#/' + data.state.indexh + '/' + data.state.indexv;\n\t\t\t\t\tvar currentURL = data.url + urlSeparator + params + '&postMessageEvents=true' + hash;\n\t\t\t\t\tvar upcomingURL = data.url + urlSeparator + params + '&controls=false' + hash;\n\n\t\t\t\t\tcurrentSlide = document.createElement( 'iframe' );\n\t\t\t\t\tcurrentSlide.setAttribute( 'width', 1280 );\n\t\t\t\t\tcurrentSlide.setAttribute( 'height', 1024 );\n\t\t\t\t\tcurrentSlide.setAttribute( 'src', currentURL );\n\t\t\t\t\tdocument.querySelector( '#current-slide' ).appendChild( currentSlide );\n\n\t\t\t\t\tupcomingSlide = document.createElement( 'iframe' );\n\t\t\t\t\tupcomingSlide.setAttribute( 'width', 640 );\n\t\t\t\t\tupcomingSlide.setAttribute( 'height', 512 );\n\t\t\t\t\tupcomingSlide.setAttribute( 'src', upcomingURL );\n\t\t\t\t\tdocument.querySelector( '#upcoming-slide' ).appendChild( upcomingSlide );\n\n\t\t\t\t\twhitelistedWindows.push( currentSlide.contentWindow, upcomingSlide.contentWindow );\n\n\t\t\t\t}\n\n\t\t\t\t/**\n\t\t\t\t * Setup the notes UI.\n\t\t\t\t */\n\t\t\t\tfunction setupNotes() {\n\n\t\t\t\t\tnotes = document.querySelector( '.speaker-controls-notes' );\n\t\t\t\t\tnotesValue = document.querySelector( '.speaker-controls-notes .value' );\n\n\t\t\t\t}\n\n\t\t\t\t/**\n\t\t\t\t * We send out a heartbeat at all times to ensure we can\n\t\t\t\t * reconnect with the main presentation window after reloads.\n\t\t\t\t */\n\t\t\t\tfunction setupHeartbeat() {\n\n\t\t\t\t\tsetInterval( () => {\n\t\t\t\t\t\twindow.opener.postMessage( JSON.stringify({ namespace: 'reveal-notes', type: 'heartbeat'} ), '*' );\n\t\t\t\t\t}, 1000 );\n\n\t\t\t\t}\n\n\t\t\t\tfunction getTimings( callback ) {\n\n\t\t\t\t\tcallRevealApi( 'getSlidesAttributes', [], function ( slideAttributes ) {\n\t\t\t\t\t\tcallRevealApi( 'getConfig', [], function ( config ) {\n\t\t\t\t\t\t\tvar totalTime = config.totalTime;\n\t\t\t\t\t\t\tvar minTimePerSlide = config.minimumTimePerSlide || 0;\n\t\t\t\t\t\t\tvar defaultTiming = config.defaultTiming;\n\t\t\t\t\t\t\tif ((defaultTiming == null) && (totalTime == null)) {\n\t\t\t\t\t\t\t\tcallback(null);\n\t\t\t\t\t\t\t\treturn;\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t// Setting totalTime overrides defaultTiming\n\t\t\t\t\t\t\tif (totalTime) {\n\t\t\t\t\t\t\t\tdefaultTiming = 0;\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\tvar timings = [];\n\t\t\t\t\t\t\tfor ( var i in slideAttributes ) {\n\t\t\t\t\t\t\t\tvar slide = slideAttributes[ i ];\n\t\t\t\t\t\t\t\tvar timing = defaultTiming;\n\t\t\t\t\t\t\t\tif( slide.hasOwnProperty( 'data-timing' )) {\n\t\t\t\t\t\t\t\t\tvar t = slide[ 'data-timing' ];\n\t\t\t\t\t\t\t\t\ttiming = parseInt(t);\n\t\t\t\t\t\t\t\t\tif( isNaN(timing) ) {\n\t\t\t\t\t\t\t\t\t\tconsole.warn(\"Could not parse timing '\" + t + \"' of slide \" + i + \"; using default of \" + defaultTiming);\n\t\t\t\t\t\t\t\t\t\ttiming = defaultTiming;\n\t\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t\ttimings.push(timing);\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\tif ( totalTime ) {\n\t\t\t\t\t\t\t\t// After we've allocated time to individual slides, we summarize it and\n\t\t\t\t\t\t\t\t// subtract it from the total time\n\t\t\t\t\t\t\t\tvar remainingTime = totalTime - timings.reduce( function(a, b) { return a + b; }, 0 );\n\t\t\t\t\t\t\t\t// The remaining time is divided by the number of slides that have 0 seconds\n\t\t\t\t\t\t\t\t// allocated at the moment, giving the average time-per-slide on the remaining slides\n\t\t\t\t\t\t\t\tvar remainingSlides = (timings.filter( function(x) { return x == 0 }) ).length\n\t\t\t\t\t\t\t\tvar timePerSlide = Math.round( remainingTime / remainingSlides, 0 )\n\t\t\t\t\t\t\t\t// And now we replace every zero-value timing with that average\n\t\t\t\t\t\t\t\ttimings = timings.map( function(x) { return (x==0 ? timePerSlide : x) } );\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\tvar slidesUnderMinimum = timings.filter( function(x) { return (x < minTimePerSlide) } ).length\n\t\t\t\t\t\t\tif ( slidesUnderMinimum ) {\n\t\t\t\t\t\t\t\tmessage = \"The pacing time for \" + slidesUnderMinimum + \" slide(s) is under the configured minimum of \" + minTimePerSlide + \" seconds. Check the data-timing attribute on individual slides, or consider increasing the totalTime or minimumTimePerSlide configuration options (or removing some slides).\";\n\t\t\t\t\t\t\t\talert(message);\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\tcallback( timings );\n\t\t\t\t\t\t} );\n\t\t\t\t\t} );\n\n\t\t\t\t}\n\n\t\t\t\t/**\n\t\t\t\t * Return the number of seconds allocated for presenting\n\t\t\t\t * all slides up to and including this one.\n\t\t\t\t */\n\t\t\t\tfunction getTimeAllocated( timings, callback ) {\n\n\t\t\t\t\tcallRevealApi( 'getSlidePastCount', [], function ( currentSlide ) {\n\t\t\t\t\t\tvar allocated = 0;\n\t\t\t\t\t\tfor (var i in timings.slice(0, currentSlide + 1)) {\n\t\t\t\t\t\t\tallocated += timings[i];\n\t\t\t\t\t\t}\n\t\t\t\t\t\tcallback( allocated );\n\t\t\t\t\t} );\n\n\t\t\t\t}\n\n\t\t\t\t/**\n\t\t\t\t * Create the timer and clock and start updating them\n\t\t\t\t * at an interval.\n\t\t\t\t */\n\t\t\t\tfunction setupTimer() {\n\n\t\t\t\t\tvar start = new Date(),\n\t\t\t\t\ttimeEl = document.querySelector( '.speaker-controls-time' ),\n\t\t\t\t\tclockEl = timeEl.querySelector( '.clock-value' ),\n\t\t\t\t\thoursEl = timeEl.querySelector( '.hours-value' ),\n\t\t\t\t\tminutesEl = timeEl.querySelector( '.minutes-value' ),\n\t\t\t\t\tsecondsEl = timeEl.querySelector( '.seconds-value' ),\n\t\t\t\t\tpacingTitleEl = timeEl.querySelector( '.pacing-title' ),\n\t\t\t\t\tpacingEl = timeEl.querySelector( '.pacing' ),\n\t\t\t\t\tpacingHoursEl = pacingEl.querySelector( '.hours-value' ),\n\t\t\t\t\tpacingMinutesEl = pacingEl.querySelector( '.minutes-value' ),\n\t\t\t\t\tpacingSecondsEl = pacingEl.querySelector( '.seconds-value' );\n\n\t\t\t\t\tvar timings = null;\n\t\t\t\t\tgetTimings( function ( _timings ) {\n\n\t\t\t\t\t\ttimings = _timings;\n\t\t\t\t\t\tif (_timings !== null) {\n\t\t\t\t\t\t\tpacingTitleEl.style.removeProperty('display');\n\t\t\t\t\t\t\tpacingEl.style.removeProperty('display');\n\t\t\t\t\t\t}\n\n\t\t\t\t\t\t// Update once directly\n\t\t\t\t\t\t_updateTimer();\n\n\t\t\t\t\t\t// Then update every second\n\t\t\t\t\t\tsetInterval( _updateTimer, 1000 );\n\n\t\t\t\t\t} );\n\n\n\t\t\t\t\tfunction _resetTimer() {\n\n\t\t\t\t\t\tif (timings == null) {\n\t\t\t\t\t\t\tstart = new Date();\n\t\t\t\t\t\t\t_updateTimer();\n\t\t\t\t\t\t}\n\t\t\t\t\t\telse {\n\t\t\t\t\t\t\t// Reset timer to beginning of current slide\n\t\t\t\t\t\t\tgetTimeAllocated( timings, function ( slideEndTimingSeconds ) {\n\t\t\t\t\t\t\t\tvar slideEndTiming = slideEndTimingSeconds * 1000;\n\t\t\t\t\t\t\t\tcallRevealApi( 'getSlidePastCount', [], function ( currentSlide ) {\n\t\t\t\t\t\t\t\t\tvar currentSlideTiming = timings[currentSlide] * 1000;\n\t\t\t\t\t\t\t\t\tvar previousSlidesTiming = slideEndTiming - currentSlideTiming;\n\t\t\t\t\t\t\t\t\tvar now = new Date();\n\t\t\t\t\t\t\t\t\tstart = new Date(now.getTime() - previousSlidesTiming);\n\t\t\t\t\t\t\t\t\t_updateTimer();\n\t\t\t\t\t\t\t\t} );\n\t\t\t\t\t\t\t} );\n\t\t\t\t\t\t}\n\n\t\t\t\t\t}\n\n\t\t\t\t\ttimeEl.addEventListener( 'click', function() {\n\t\t\t\t\t\t_resetTimer();\n\t\t\t\t\t\treturn false;\n\t\t\t\t\t} );\n\n\t\t\t\t\tfunction _displayTime( hrEl, minEl, secEl, time) {\n\n\t\t\t\t\t\tvar sign = Math.sign(time) == -1 ? \"-\" : \"\";\n\t\t\t\t\t\ttime = Math.abs(Math.round(time / 1000));\n\t\t\t\t\t\tvar seconds = time % 60;\n\t\t\t\t\t\tvar minutes = Math.floor( time / 60 ) % 60 ;\n\t\t\t\t\t\tvar hours = Math.floor( time / ( 60 * 60 )) ;\n\t\t\t\t\t\thrEl.innerHTML = sign + zeroPadInteger( hours );\n\t\t\t\t\t\tif (hours == 0) {\n\t\t\t\t\t\t\thrEl.classList.add( 'mute' );\n\t\t\t\t\t\t}\n\t\t\t\t\t\telse {\n\t\t\t\t\t\t\thrEl.classList.remove( 'mute' );\n\t\t\t\t\t\t}\n\t\t\t\t\t\tminEl.innerHTML = ':' + zeroPadInteger( minutes );\n\t\t\t\t\t\tif (hours == 0 && minutes == 0) {\n\t\t\t\t\t\t\tminEl.classList.add( 'mute' );\n\t\t\t\t\t\t}\n\t\t\t\t\t\telse {\n\t\t\t\t\t\t\tminEl.classList.remove( 'mute' );\n\t\t\t\t\t\t}\n\t\t\t\t\t\tsecEl.innerHTML = ':' + zeroPadInteger( seconds );\n\t\t\t\t\t}\n\n\t\t\t\t\tfunction _updateTimer() {\n\n\t\t\t\t\t\tvar diff, hours, minutes, seconds,\n\t\t\t\t\t\tnow = new Date();\n\n\t\t\t\t\t\tdiff = now.getTime() - start.getTime();\n\n\t\t\t\t\t\tclockEl.innerHTML = now.toLocaleTimeString( 'en-US', { hour12: true, hour: '2-digit', minute:'2-digit' } );\n\t\t\t\t\t\t_displayTime( hoursEl, minutesEl, secondsEl, diff );\n\t\t\t\t\t\tif (timings !== null) {\n\t\t\t\t\t\t\t_updatePacing(diff);\n\t\t\t\t\t\t}\n\n\t\t\t\t\t}\n\n\t\t\t\t\tfunction _updatePacing(diff) {\n\n\t\t\t\t\t\tgetTimeAllocated( timings, function ( slideEndTimingSeconds ) {\n\t\t\t\t\t\t\tvar slideEndTiming = slideEndTimingSeconds * 1000;\n\n\t\t\t\t\t\t\tcallRevealApi( 'getSlidePastCount', [], function ( currentSlide ) {\n\t\t\t\t\t\t\t\tvar currentSlideTiming = timings[currentSlide] * 1000;\n\t\t\t\t\t\t\t\tvar timeLeftCurrentSlide = slideEndTiming - diff;\n\t\t\t\t\t\t\t\tif (timeLeftCurrentSlide < 0) {\n\t\t\t\t\t\t\t\t\tpacingEl.className = 'pacing behind';\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t\telse if (timeLeftCurrentSlide < currentSlideTiming) {\n\t\t\t\t\t\t\t\t\tpacingEl.className = 'pacing on-track';\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t\telse {\n\t\t\t\t\t\t\t\t\tpacingEl.className = 'pacing ahead';\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t\t_displayTime( pacingHoursEl, pacingMinutesEl, pacingSecondsEl, timeLeftCurrentSlide );\n\t\t\t\t\t\t\t} );\n\t\t\t\t\t\t} );\n\t\t\t\t\t}\n\n\t\t\t\t}\n\n\t\t\t\t/**\n\t\t\t\t * Sets up the speaker view layout and layout selector.\n\t\t\t\t */\n\t\t\t\tfunction setupLayout() {\n\n\t\t\t\t\tlayoutDropdown = document.querySelector( '.speaker-layout-dropdown' );\n\t\t\t\t\tlayoutLabel = document.querySelector( '.speaker-layout-label' );\n\n\t\t\t\t\t// Render the list of available layouts\n\t\t\t\t\tfor( var id in SPEAKER_LAYOUTS ) {\n\t\t\t\t\t\tvar option = document.createElement( 'option' );\n\t\t\t\t\t\toption.setAttribute( 'value', id );\n\t\t\t\t\t\toption.textContent = SPEAKER_LAYOUTS[ id ];\n\t\t\t\t\t\tlayoutDropdown.appendChild( option );\n\t\t\t\t\t}\n\n\t\t\t\t\t// Monitor the dropdown for changes\n\t\t\t\t\tlayoutDropdown.addEventListener( 'change', function( event ) {\n\n\t\t\t\t\t\tsetLayout( layoutDropdown.value );\n\n\t\t\t\t\t}, false );\n\n\t\t\t\t\t// Restore any currently persisted layout\n\t\t\t\t\tsetLayout( getLayout() );\n\n\t\t\t\t}\n\n\t\t\t\t/**\n\t\t\t\t * Sets a new speaker view layout. The layout is persisted\n\t\t\t\t * in local storage.\n\t\t\t\t */\n\t\t\t\tfunction setLayout( value ) {\n\n\t\t\t\t\tvar title = SPEAKER_LAYOUTS[ value ];\n\n\t\t\t\t\tlayoutLabel.innerHTML = 'Layout' + ( title ? ( ': ' + title ) : '' );\n\t\t\t\t\tlayoutDropdown.value = value;\n\n\t\t\t\t\tdocument.body.setAttribute( 'data-speaker-layout', value );\n\n\t\t\t\t\t// Persist locally\n\t\t\t\t\tif( supportsLocalStorage() ) {\n\t\t\t\t\t\twindow.localStorage.setItem( 'reveal-speaker-layout', value );\n\t\t\t\t\t}\n\n\t\t\t\t}\n\n\t\t\t\t/**\n\t\t\t\t * Returns the ID of the most recently set speaker layout\n\t\t\t\t * or our default layout if none has been set.\n\t\t\t\t */\n\t\t\t\tfunction getLayout() {\n\n\t\t\t\t\tif( supportsLocalStorage() ) {\n\t\t\t\t\t\tvar layout = window.localStorage.getItem( 'reveal-speaker-layout' );\n\t\t\t\t\t\tif( layout ) {\n\t\t\t\t\t\t\treturn layout;\n\t\t\t\t\t\t}\n\t\t\t\t\t}\n\n\t\t\t\t\t// Default to the first record in the layouts hash\n\t\t\t\t\tfor( var id in SPEAKER_LAYOUTS ) {\n\t\t\t\t\t\treturn id;\n\t\t\t\t\t}\n\n\t\t\t\t}\n\n\t\t\t\tfunction supportsLocalStorage() {\n\n\t\t\t\t\ttry {\n\t\t\t\t\t\tlocalStorage.setItem('test', 'test');\n\t\t\t\t\t\tlocalStorage.removeItem('test');\n\t\t\t\t\t\treturn true;\n\t\t\t\t\t}\n\t\t\t\t\tcatch( e ) {\n\t\t\t\t\t\treturn false;\n\t\t\t\t\t}\n\n\t\t\t\t}\n\n\t\t\t\tfunction zeroPadInteger( num ) {\n\n\t\t\t\t\tvar str = '00' + parseInt( num );\n\t\t\t\t\treturn str.substring( str.length - 2 );\n\n\t\t\t\t}\n\n\t\t\t\t/**\n\t\t\t\t * Limits the frequency at which a function can be called.\n\t\t\t\t */\n\t\t\t\tfunction debounce( fn, ms ) {\n\n\t\t\t\t\tvar lastTime = 0,\n\t\t\t\t\t\ttimeout;\n\n\t\t\t\t\treturn function() {\n\n\t\t\t\t\t\tvar args = arguments;\n\t\t\t\t\t\tvar context = this;\n\n\t\t\t\t\t\tclearTimeout( timeout );\n\n\t\t\t\t\t\tvar timeSinceLastCall = Date.now() - lastTime;\n\t\t\t\t\t\tif( timeSinceLastCall > ms ) {\n\t\t\t\t\t\t\tfn.apply( context, args );\n\t\t\t\t\t\t\tlastTime = Date.now();\n\t\t\t\t\t\t}\n\t\t\t\t\t\telse {\n\t\t\t\t\t\t\ttimeout = setTimeout( function() {\n\t\t\t\t\t\t\t\tfn.apply( context, args );\n\t\t\t\t\t\t\t\tlastTime = Date.now();\n\t\t\t\t\t\t\t}, ms - timeSinceLastCall );\n\t\t\t\t\t\t}\n\n\t\t\t\t\t}\n\n\t\t\t\t}\n\n\t\t\t})();\n\n\t\t</script>\n\t</body>\n</html>",
          ),
          !n)
        )
          return void alert(
            "Speaker view popup failed to open. Please make sure popups are allowed and reopen the speaker view.",
          );
        (r = e.getConfig().url),
          (i =
            "string" == typeof r
              ? r
              : window.location.protocol +
                "//" +
                window.location.host +
                window.location.pathname +
                window.location.search),
          (t = setInterval(function () {
            n.postMessage(
              JSON.stringify({
                namespace: "reveal-notes",
                type: "connect",
                state: e.getState(),
                url: i,
              }),
              "*",
            );
          }, 500)),
          window.addEventListener("message", u);
      }
      var r, i;
    }
    function i(t) {
      var r = e.getCurrentSlide(),
        i = r.querySelector("aside.notes"),
        u = r.querySelector(".current-fragment"),
        a = {
          namespace: "reveal-notes",
          type: "state",
          notes: "",
          markdown: !1,
          whitespace: "normal",
          state: e.getState(),
        };
      if (
        (r.hasAttribute("data-notes") &&
          ((a.notes = r.getAttribute("data-notes")),
          (a.whitespace = "pre-wrap")),
        u)
      ) {
        var o = u.querySelector("aside.notes");
        o
          ? (i = o)
          : u.hasAttribute("data-notes") &&
            ((a.notes = u.getAttribute("data-notes")),
            (a.whitespace = "pre-wrap"),
            (i = null));
      }
      i &&
        ((a.notes = i.innerHTML),
        (a.markdown = "string" == typeof i.getAttribute("data-markdown"))),
        n.postMessage(JSON.stringify(a), "*");
    }
    function u(r) {
      var i,
        u,
        o,
        s,
        l = JSON.parse(r.data);
      l && "reveal-notes" === l.namespace && "connected" === l.type
        ? (clearInterval(t), a())
        : l &&
          "reveal-notes" === l.namespace &&
          "call" === l.type &&
          ((i = l.methodName),
          (u = l.arguments),
          (o = l.callId),
          (s = e[i].apply(e, u)),
          n.postMessage(
            JSON.stringify({
              namespace: "reveal-notes",
              type: "return",
              result: s,
              callId: o,
            }),
            "*",
          ));
    }
    function a() {
      e.on("slidechanged", i),
        e.on("fragmentshown", i),
        e.on("fragmenthidden", i),
        e.on("overviewhidden", i),
        e.on("overviewshown", i),
        e.on("paused", i),
        e.on("resumed", i),
        i();
    }
    return {
      id: "notes",
      init: function (t) {
        (e = t),
          /receiver/i.test(window.location.search) ||
            (null !== window.location.search.match(/(\?|\&)notes/gi)
              ? r()
              : window.addEventListener("message", function (t) {
                  if (!n && "string" == typeof t.data) {
                    var e;
                    try {
                      e = JSON.parse(t.data);
                    } catch (t) {}
                    e &&
                      "reveal-notes" === e.namespace &&
                      "heartbeat" === e.type &&
                      ((r = t.source),
                      n && !n.closed
                        ? n.focus()
                        : ((n = r),
                          window.addEventListener("message", u),
                          a()));
                  }
                  var r;
                }),
            e.addKeyBinding(
              { keyCode: 83, key: "S", description: "Speaker notes view" },
              function () {
                r();
              },
            ));
      },
      open: r,
    };
  };
});
