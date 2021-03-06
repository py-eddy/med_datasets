CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?h?t?j~?   max       ??j~??"?      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M?k?   max       P?!      ?  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ?t?   max       =?S?      ?  ?   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?=p??
   max       @F?=p??
     	?   ?   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??z?G??   max       @vw\(?     	?  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @/         max       @L?           ?  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @?         max       @??`          ?  4?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ????   max       >2-      ?  5?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A??^   max       B4??      ?  6?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?{Q   max       B4??      ?  7?   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ??]?   max       C???      ?  8?   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ???b   max       C??e      ?  9?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          p      ?  :?   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      ?  ;?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      ?  <?   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M?k?   max       P3?w      ?  =?   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ???D??*   max       ?؋C??%      ?  >?   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ?t?   max       =?`B      ?  ??   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>????R   max       @F?=p??
     	?  @?   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ???Q???   max       @vw\(?     	?  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @)         max       @L?           ?  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?         max       @???          ?  T?   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @?   max         @?      ?  U?   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ?r??n/   max       ??w?kP??     ?  V?   
                  	   !   	      $   b            	         N      h   7   M         &         	   +            6      .   
   N      B      (   %   
      )   k      +   #   +            o                     4   N9i?M?k?N?U?Oð&Nm?VO'C9N?3=O?07N??uN??O?O?\BN6	O??pN??O&iO<ݫO??P???O??P?!O???PFF?O???O?o?P0??NG?N'??O0Y?Pk̫Or&?Ns??O[)P?NJ"?O?9N2KP_K"N??)P$??NG/eO??P?GNق3N?TEO?qP??N.?O?rqO?<	O???N[?|O?$7N?GPMNiNX??N?anO2?bN{??NP??N?7$N?L^?t????
?D????o?o?o?o?o;?o;??
;ě?;?`B;?`B<o<#?
<#?
<49X<49X<T??<T??<T??<e`B<?C?<?C?<?C?<?C?<?C?<?t?<???<???<?1<?1<?1<?9X<?j<ě?<???<?`B<?`B<?<?<???=+=\)=\)=t?=0 ?=0 ?=49X=49X=49X=P?`=P?`=e`B=?%=?+=?O?=??=???=? ?=??`=?"?=?S?")5BIGB5/)""""""""""????????????????????gdeegty??????tsggggg??????????????????
$"%??????????????????????????


???????
#/<?OQOH<4#??#$'0<>==<40#????????????????????>;DK[h???????uh_[OB>=89<CN[gt?????~tg[F=????????????????????ct??????????????tgc???????????????????????????	"/;<<3100/("
egt???????????????te????????)481'????JIJT[cmt????????tg[Jl|????2>:+????znl#/<HUoqnk`UC</$????$5SZXB)?????/HUahmjkaUH<.##/<DSWUSKH</#gs???????????????zg????????????????????:12<HKLH?<::::::::::%)5=854640)HCQg??????????????jH??????????????????????????????????????????????????????????????????9FNQRPE???????

??????????$%)014B[gt??ygNB5*$sptz???????tssssssss???)BNZcb^M@)???#07560#??)5N_ef_XNB5)?zx????????zzzzzzzzzz?????????????????????????)39451&??%*/7<HNUYWUSIHG<2/%%E?HUannxnaUHEEEEEEEE??????????????????????? ')??????????????????????????+()*-3;AHT^ghc`UH;/+????????????????????????????????????????ckn{????{ncccccccccc??????????????ost|?????toooooooooo????)6BGIB;6$????	

 
										\Yamzz?zma\\\\\\\\\\???????????????????????????????????????????????????????????}|???????}}}}}}}}}}??????

??????ST[`htv?????thh[SSSS???????????????????????????????????????뺤??????????????????????????????????????Óàìùþ??ùììàÓÇÄÆÇÒÓÓÓÓ????4?E?K?F?A?4?(????߽ݽҽӽڽ??????(?4?4?4?1?(??????????????????????????????????????????????????????????)?+?*?)?????? ??????? ???`?m?}?{?|?y?`?T?.?"?	????	??"?.?G?]?`???????????????????????????}?y?}???????????????????????????????????????????????ҾA?M?f?s??????????????r?c?Z?M?A?4?A?6?A?6?B?O?[?h?r?v?w?s?h?[?O?G??????)?6?[?a?h?k?h?^?[?O?E?N?O?T?[?[?[?[?[?[?[?[?y?????????ǿʿĿ????y?u?r?k?k?q?y?s?s?yù??????????????ùíììàÙßàìðùù??"?+?.?8?3?.?*?"???	????????????????T?a?m?v?y?p?m?a?H?;?/?"??"?%?/?7?;?H?T????????????????????????s?e?e?j?j?z?????????/?H?m???????????????m?T?H?4?	????????????/?7?;?H?U?Z?H?;?/?????????????????????/?B?O?R?J?;?"?	??????????????????????????!?#?!???????????????????????T?e???????????????m?G?;?+?&?)?9?C?C?G?T?????????ľƾž???????s?f?_?U?N?f?w????ݿ???	???????????ݿĿ??????Ŀѿڿݿ???(?Z?g???????????s?g?N?????????????׾????????????????׾վϾ׾׾׾׾׾׾׾?£?u?????#?<?A?H?G?A?5?0?(??????????????A?N?g?z???????????s?f?d?Z?A?????(?A?\?h?uƁƍƚƨƪƧƠƚƎƁ?\?O?F?F?O?Z?\E?E?E?E?E?FFFFFFE?E?E?E?E?E?E?E?E??4?M?Z?_?c?r?~?s?h?M?A?4?(??????(?4???ʾ??	?"?4?;?<?9?.?"??	?????׾ɾ??????????ʼμӼͼʼɼ????????????????????????)?5?B?N?[?g?b?a?`?Z?P?B?)???????)ù????????????ùìêìøùùùùùùùù?uƚƳ??????????????ƚƁ?c?X?K?O?\?u????????? ????????ۼؼݼ??????????????????????!?$?"??
??????¿¥¸???俸?ĿѿѿֿѿĿ?????????????????????????????????*?6?:?=???6?*??????????????????????#?0?b?n?|ŀ?~?n?U?<?0?????????????n?zÀÇÊÌÇ?~?z?n?e?a?U?S?U?_?a?b?n?n???$?"??????????????????????????"?,?3?6?2?"??	???????????????????????????޻û??????l?e?d?q?z????????ìù????ùòìàßàêëìììììììì??????????#?-?0?2?.?#??
??????????????????)?-?0?)???????????????????????tāąĐİĶĶĦč?h?[?R?J?B?7?F?J?O?[?t????????????????????????????????????y???????????????????????y?l?e?g?f?[?l?y??!?&?-?/?-?!???
???????????r???????ź˺Һպɺ??????r?L?'??? ?3?r??????????????????????????????????????߿ĿѿؿٿѿѿĿ??????ĿĿĿĿĿĿĿĿĿļ????????????????????ۼټ޼??????????????#?/?<?H?U?W?X?V?U?H?F?<?/?#????? ?#?/?<?D?F?H?J?H?<?/?.?#?!?#?&?/?/?/?/?/?/?4?@?M?R?Y?M?@?4?)?.?4?4?4?4?4?4?4?4?4?4D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D????????????????????????????????????????? _ X I ) G ? ; b Y F _ & o H N A U l f M Y + @  ) O = 9 c R E R V * > ( . \ + 4 9 [ b , h O R j 4 K G h 4 8 X G F s ) @ 7 9 B    ~     ?  ?  ?    ?    ?  ?  O  ?  ?  ?  ?  n  ?  n  ?  )  C  \  ~  
  ?  |  g  :  ?       ?    ?  m  ?  I  0  ?  ?  f  ?  M  ?  ?  N  ?  O  Y  ?  ?  ?  C  *  ?  9  p  7  ?  ~  _    ??????u;o<?/<?C?<49X;ě?<???<D??<?C?='??=?
=<49X=C?<?1<???<??h<?j=?Q?=??==?O?=??=@?=t?=]/<??
=o<?/=y?#=,1<?`B=#?
=???<?`B=?\)=C?=??=+=ě?=+=?hs=?\)=8Q?=0 ?=??->?u=P?`=? ?=???=? ?=?+=??=u>2-=?hs=???=?-=?j=ě?=?>"??>$?B;B"#B	?B ?"B	FBvuBo?B?B%itBܜB-?B	+BB?RBҜB ??A??^B??B??B	?BXBڃB??B?B\?B??B4??B??Bz?Bj?B??B??B?oB?0B#?yB?bB??B?SB%R?B+@BvBe?B??B??B??Bd?BQB!??A?~B?!B?B(?[B- $B??B?BYA?o?BBc?B?*B?B\?B?HBB?B"?ZB	??B ??B??BuBEB??B%??B?NB?B	?4BN<B??B??B ??A?{QBy?B?}B	??B?B??B??B??B?XB?!B4??B?GB5GB;?B? B?B??B?8B#H?B?1B?
B??B%ADB0PBB<?B??B<?B?HB? B?eB!?<A??B6?B??B(GJB,ΙB?BA?B?A?y?B??B?WB??B>B@B??A?<@%cA?r?A3?A5v?A???A?u?Ac?:@??A?HAF6?A?x?A?qAr#EA?A\"?A???AG??A?
QA?aOA??AҬAi?SAG?7A~r?A??{AU?A??A?XqA???B?VC???A;k?AY4P@?SA?f?A??B5KA?A???Ax?rA???A???A?͐A3?A?bd@?[?A̼?A??*A??pA?u?@V?SA?~@hj???]?A???Ay?tAGUA???A?8@?bC??e@?܁AЈl@??A˄?A3S?A3yA?j?A?iAct@???A?{6AG8?A؀Aځ?As?A͖KA[A?R?AE?A?O^A?^?A?M?A?|?Ai?AGjiA}A???AU?A???A?%?A?)?B??C??eA<??AY+@?R?A??#A?p?B??A?A??AyyA?mA??dA?\.A3N?A??a@?آA??A???A?~?A?b@S?:A??@d;????bA??!Ay??A
?A?z?A?@??TC??@???                      	   "   	      $   c            
         N      i   8   N         &         	   +            6      /   
   O      C      )   &      	   *   k   	   +   $   +            p                     5               !            #         +   !      #            #   9   %   ?      /   !      1            5            '      !      5      )         +         %   )            #            3                                                                              #   )   #   %      !   !                  -                              #         +         #                           '                        N9i?M?k?N]SOz/?Nm?VO?3N?3=O`ihN??uN\$-OP?IO?KxN6	O???NoͲO&iOo\O??P3?O??P
?RN?e$O??O???O?o?N?]NG?N'??O0Y?P3?wOR:?Ns??O[)O?NJ"?O^YYN2KO?FN??)O?`qNG/eOP#?P?GN???N?TEO?|[O??'N.?OS;?O??aO?k?N[?|O?$7N?GP ?+NiNX??N?anO2?bN{??NP??N??N???  t  [  c  B  ^      ;  ?  ?  4  ?  `  K  ?  4  ?  v  .  6  	C  :  ?  G  p    I  ?  ?  /    i  ?  6  ?  ?  ?  ?  ?  ?  ?  \    L  ?  :  ?  O  ?  Q    [    ?    H  ?  ?  ?  F  1  ?  ?t????
?o;?o?o??o?o;?`B;?o;?`B<???=\);?`B<D??<T??<#?
<e`B<49X=t?<?o=<j=?P=@?<?C?<?C?=#?
<?C?<?t?<???<???<?j<?1<?1=t?<?j=??<???=??<?`B=,1<?='??=+=t?=\)='??=?\)=0 ?=Y?=8Q?=Y?=P?`=P?`=e`B=?j=?+=?O?=??=???=? ?=??`=?S?=?`B")5BIGB5/)""""""""""????????????????????gght|????tlggggggggg????????????????????
$"%??????????????????????????


?????
#/4<@HJKJH</##$'0<>==<40#????????????????????UWWZ^bht????????th[UACGNP[gt}???~ytg[NGA????????????????????z}?????????????????z???
????????????????????????????
"/8951/+"egt???????????????te???????'*(!?????MKLQ[jt?????????ytgM??????)/0-??????*''-/<HITUYUTHC<7/**?????):AEA5)???/HUahmjkaUH<.##/<DSWUSKH</#????????????????????????????????????????:12<HKLH?<::::::::::%)5=854640)YR\g??????????????qY?????????????????????????????????????????????????????????????????6EHIHD6)???????

??????????36:BN[grtzzxtpg[NB<3sptz???????tssssssss		)57BDB>5)%#07560#
 5BNY`^[SHB5zx????????zzzzzzzzzz?????????????????????????)39451&??',/9<HMUWVUQH<;/''''E?HUannxnaUHEEEEEEEE????????	????????????????
????????????????????????;20013:;HTXaca^YTH;;????????????????????????????????????????ckn{????{ncccccccccc??????????????ost|?????toooooooooo??????)00.+&????	

 
										\Yamzz?zma\\\\\\\\\\???????????????????????????????????????????????????????????}|???????}}}}}}}}}}??????

????????ST[`htu?????tjh[SSSS???????????????????????????????????????뺤??????????????????????????????????????àæì÷ìèàÓÌÌÓÝàààààààà????(?4?=?E?>?4?(???????????????????(?4?4?4?1?(??????????????????????????????????????????????????????????)?+?*?)?????? ??????? ???T?`?m?q?o?k?`?S?G?4?.?"?????#?.?;?T???????????????????????????}?y?}???????????????????????????????????????????????Ҿf?s???????????????????????s?p?f?\?\?f?B?O?[?g?h?k?l?h?[?O?B?6?)?????)?6?B?[?a?h?k?h?^?[?O?E?N?O?T?[?[?[?[?[?[?[?[???????????ÿſĿ??????????y?s?s?u?x????ù??????????????ùõìåìôùùùùùù??"?+?.?8?3?.?*?"???	????????????????H?T?a?k?m?t?u?m?a?T?H?@?;?1?;?=?H?H?H?H????????????????????????s?e?e?j?j?z?????;?H?a?m???????????????m?a?T?H?,???!?;??????"?3?;?K?Q?S?H?;?/?	?????????????????	??/?;?E?E?<?/?"??	?????????????????????????????????????????????????`?m?y?????????|?m?`?T?G?;?9?9?<?B?J?T?`?????????ľƾž???????s?f?_?U?N?f?w????ݿ???	???????????ݿĿ??????Ŀѿڿ??N?Z?b?g?m?n?g?f?Z?N?B?A?5?3?0?5?A?E?N?N?׾????????????????׾վϾ׾׾׾׾׾׾׾?£?u?????#?<?A?H?G?A?5?0?(??????????????A?N?g?p?}???????s?e?\?A??????%?(?A?h?uƁƈƚƤƧƧƝƚƎƁ?h?\?R?O?I?Q?\?hE?E?E?E?E?FFFFFFE?E?E?E?E?E?E?E?E??4?M?Z?_?c?r?~?s?h?M?A?4?(??????(?4?????	??"?-?1?.?$??	???????پվξ;վ㼱???ʼμӼͼʼɼ????????????????????????)?5?B?M?Q?T?T?N?M?B?5?)?!???????)ù????????????ùìêìøùùùùùùùùƎƚƧƳ????????????????ƳƧƓƃƁ?zƂƎ????????? ????????ۼؼݼ??????????????????????????
?? ? ???
??????¿´£«?˿??ĿѿѿֿѿĿ???????????????????????????????????'?*?3?5?0?*????????????????????#?0?b?n?|ŀ?~?n?U?<?0?????????????n?z?~ÇÈËÇ?}?z?n?k?a?V?a?a?e?n?n?n?n???$?"????????????????????????"?0?3?.?"??	?????????????????˻????ûлܻ????޻û??????????x?p?z??????ìù????ùòìàßàêëìììììììì?????????
??#?%?,?'?#??
??????????????????+?-?)?????????????????????????jāčĚĪİĭĦĚčā?t?h?_?S?P?S?[?f?j????????????????????????????????????y???????????????????????y?l?e?g?f?[?l?y??!?&?-?/?-?!???
???????????@?Y?r?????????????????~?r?e?@?*?%?'?3?@??????????????????????????????????????߿ĿѿؿٿѿѿĿ??????ĿĿĿĿĿĿĿĿĿļ????????????????????ۼټ޼??????????????#?/?<?H?U?W?X?V?U?H?F?<?/?#????? ?#?/?<?D?F?H?J?H?<?/?.?#?!?#?&?/?/?/?/?/?/?4?@?M?R?Y?M?@?4?)?.?4?4?4?4?4?4?4?4?4?4D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D????????????????????????????????????????? _ X 7 ( G > ; \ Y G ;  o & B A / l G P =  9  )  = 9 c O D R V " > & . G + , 9 C b 2 h V B j 4 L G h 4 8 I G F s ) @ 7 1 B    ~     `  ?  ?  [  ?    ?  {  ?  ?  ?    ?  n  %  n  ?  ?  ?    ?  
  ?    g  :  ?  A  ?  ?    ?  m  ?  I  G  ?    f  ?  M  ?  ?    ?  O  ?  w  5  ?  C  *  t  9  p  7  ?  ~  _  ?  ?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  t  ?  ?  ?  ?  ?  ?    &  F  `  s  y  w  r  m  g  _  V  L  [  S  K  B  6      ?  ?  ?  ?  ?  z  e  P  ;  &    ?  ?  T  U  X  ]  `  b  \  P  B  /      ?  ?  ?  ?  Y    ?  ?  ?      3  ?  A  5  #    ?  ?  ?  ?  e  6  ?  v  ?  <    ^  ?  k  ?  ?  ?    C  u  ?  a  ,  ?  ?  }  ?  ?  ?  t  +        ?  ?  ?  ?  ?  ?  ?  y  [  :    ?  ?  ~  /  ?  ?        ?  ?  ?  ?  ?  ?  ?  s  Z  ?    ?  ?  ?  }  W  1  ?    ,  1  9  ;  1      ?  ?  ?  V  +  ?  ?  2  ?  ?    ?  ?  ?  x  d  P  B  4  )      ?  ?  ?  ?  y  G     ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  b  >    ?  ?  ?    #  *  0  4  /  "    ?  ?  w  %  ?  j    ?  ?  
?    	  w  ?  ?  ?  ?  V    ?  J  
?  
n  	?  	      ?  ?  `  V  K  A  7  ,  "        ?  ?  ?  ?  ?  ?  u  O  *          J  B  :  5  .  $      ?  ?  ?  b  -  ?  ?  E  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  r  b  L  2    ?  ?  ?  ?  y  $  4  2  0  *  $        ?  ?  ?  ?  ?  ?  g  J  1      "  t  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  r  F    ?  ?  m    ?  ?  v  t  o  f  V  D  3  &    .  3  +    ?  ?  ?  e  C  )      e  ?  ?  ?    ,    ?  ?  ?  w  -  ?  ?  @  ?      ?  )  2  5  4  .      ?  ?  ?  o  `  <      ?  ?  }  I  ?     ?  K  ?  	  	/  	C  	1  	  ?  ?  ?  ?  C  ?    ?  ?  H  m  ,  ?    z  ?  ?    1  9  *  ?  ?  }  )  ?  @  ?    c  ?  ?  A  ?    2  W  u  ?  ?  n  F    ?  n  ?  5  k  \    ?  G  E  A  :  4  -  '      ?  ?  ?  ?  U    ?  }    ?  ?  p  f  Z  I  4    ?  ?  ?  ?  T    ?  ?  w  =  ?  ?  u    ?    ?  ?  ?  ?  ?  ?  ?  ?  ?        ?  ?  y    ?  ?  I  F  D  B  @  >  <  6  /  (  !      
   ?   ?   ?   ?   ?   ?  ?  ?  ?  ?  ?  ?  y  f  \  Q  E  8  &    ?  ?  ?  ]  C  N  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  c  I  .    ?  ?  ?  ?  N    ?  ?  (  .  $    ?  ?  ?  ?  ?  ]    ?  ~    ?  ?    ?  ?            ?  ?  ?  x  =    ?  ?  ?  a  B  #    ?  i  \  P  C  5  (         ?  ?  ?  ?  ?  ?  d  H  R  k  ?  ?  ?  ?  ?  d  F  ,  7  :  3  .  #    ?  ?  o  G    ?  ?  ?      (  2  6  /  !    ?  ?  ~  G    ?  b  ?  -  b  M  ?  ?  ?  ?  ?  ?  ?  ?    o  `  P  ?  -      ?  ?  ^  "  ?    Q  m  |  ?  ?  ?  {  o  \  @    ?  ?  0  ?  -  {  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  o  W  ?  &    8  P  l  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  f  ?  Z  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  p  g  ~  ?  ?  ?  ?  ?  ?  ?  ~  ]  B  #  ?  ?  9  ?  ?  ?  F  1  ?  ?  ?  ?  ?  ?    x  p  h  `  X  Q  H  >  4  *         ?  
    ,  R  \  R  C  -    ?  ?  c     ?  '  ?  S  s  L    ?  ?  ?  ?  ]  0     ?  ?  ?  ?  T     ?  ?  s    ?  e  L  L  L  H  B  8  +      ?  ?  ?  ?  c  5    ?  ?  j  7  ?  ?  ~  w  r  m  j  f  ^  V  L  A  6  &    	  ?  ?  ?  ?  &  2  :  5  "    ?  ?  ?  ?  ?  ?  w  /  ?  ?  J  ?  ?  \  ?    U  ?  ?  ?  ?  w  R    ?  a  
?  
;  	?  ?  ?  ?  &  ]  O  =  +      ?  d  ?  ?  u  =    ?  ?  `  $   ?   ?   k   ,    N  s  ?  ?  ?  ?  o  T  5    ?  ?  ?  9  ?  ?  ?  ?  ?  K  L  4    ?  ?  {  ?    m  L  "  ?  ?  ?  I  ?  ?    1  ?  ?  ?        ?  ?  ?  ?  E  ?  ?  e  ?  ?  "  ?  H  =  [  >  !    ?  ?  ?  ~  X  1    ?  ?  ?  x  Q  ,  ?  J  ?      
          ?  ?  ?  ?        ?  ?  M  ?  ?  -  ?  ?  ?  ?  ?  ?  ?  ?  ?    v  n  e  ]  T  E  5  $      
?  6  ?  ?  ?    ?  ?  ?  S  
?  
e  
   	?  	;  ?  ?  ?  ?  ?  H  7  '      ?  ?  ?  ?  ?  ?  ~  j  W  D  2       ?  ?  ?  ?  ?  ?  ?  ?  ?  x  h  X  G  6  #    ?  ?  ?  ?  ?  ?  ?  ?  v  X  6    ?  ?  ?  o  B    ?  ?  |  [  C    ?  ?  ?  ?  ?  ?  s  ]  B  $    ?  ?  ?  ?  o  M  /    ?       F  2      ?  ?  ?  ?  ?  ?  v  a  F  (     ?  ?  s  =    1      ?  ?  ?  ?  ?  {  [  9    ?  ?  ?  R    ?  ?  ]  m  ?  ?  ?  ?  ?  O  ?  ?  ?  V  ?  
?  	?  ?  ?  ?  Q  ?  v          ?  ?  ?  q  ;    ?  ?  7  ?  ?  F  ?  ~  ?  /