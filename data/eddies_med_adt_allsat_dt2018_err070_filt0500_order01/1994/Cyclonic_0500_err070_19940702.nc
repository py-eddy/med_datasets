CDF       
      obs    R   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ????         	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?`bM????   max       ????"??`     H  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M?4   max       P|??     H  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ??^5   max       <ě?     H   <   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>h?\)   max       @F??\)     ?  !?   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ???????    max       @v~?Q??     ?  .T   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @.         max       @P?           ?  ;$   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @??@         H  ;?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ?!??   max       <?9X     H  =   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A???   max       B5?%     H  >X   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A??\   max       B5?     H  ??   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >???   max       C??#     H  @?   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >R>?   max       C??^     H  B0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          z     H  Cx   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;     H  D?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;     H  F   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M?4   max       P|??     H  GP   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??!-w1?   max       ??s?PH     H  H?   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ??v?   max       <ě?     H  I?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>k??Q?   max       @F??\)     ?  K(   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ???????    max       @v}?Q??     ?  W?   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @$         max       @P?           ?  d?   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @??@         H  el   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?}   max         ?}     H  f?   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ???????A   max       ??qu?!?S     ?  g?                                 !   /      	      	                     	               +            ?         1      5   &   	                  6      ,                              &      y                     8      #         !   4                     (         N?DOX Neu+NS)WN?(.Oګ?N[|:O0?)M???O)*?Od?UP|??O?N?Z^N???NB?EP!J?N??O?T?N???N?vAOB?#N???O?)Ox?BNaENR?LO?3eOU /OEK?OZ??O?eKN??#N/m?O?Q?O?3pO?D?O?Y
Nb??NG?8M?4N?x?O???O?!P?GN?sP?;NAp?O8??N|N?O} 5NC??NK^1OC??N+#dO ueOC? N?C?P2?0O?fO_?N{	=N.??O26sO?:?P'WN?O??N??O?>-O.O?PN?3?N?TO??O ??N^??O
,'O??Nh?O(??Oq<ě?<?1<49X;?`B:?o:?o%   ??o?o??o???
?ě??o?t??t??D???T???u?u??t???t???t????㼛?㼛?㼛?㼛?㼣?
???
???
??1??1??9X??9X??9X??9X??9X??j??j?ě????????????????h???h???????????o?o?o?o?C??C??t??t??t???P??P??P?,1?,1?0 Ž0 Ž@??T???T???]/?aG??aG??ixսixսq???y?#???
???罬1??1??1??^5??^5jnsz??{{zxnmjjjjjjjj???????	?????????mmovz{?????zmlmmmmmm2<=HQRSH<:2222222222_aimz?}zwmaaXU______??)5@C?E@5)????kmrz??zumkjkkkkkkkk????????????????NO[htvtjh[QONNNNNNNN%)/13.	???FHUanz???????zna\PHF1/05Tamz????|yTH;/+1#(,/3452/#????????????????????????????????????)/016<FGC=</))))))))?????????????????????????????????????????)59?B?5)????????????????????????359:BNVWWTQNJB>75.33#+6CELLKGC=60*#???????????????????????????????????????????)*)???????!)+..1)%????????????????????y}???????????????~zy?????????????gkt???????????tkhcg?????????

???????????????????????????357BNWUPNB>50/333333JNQ[ghig`[TNJJJJJJJJgt???????????g]]\\_g?????#/4/+#
????? /<Uafqhd^\UH<8/???????????????????? )6BLNB6)#          ??????????????????????????????????rt??????????????{trrhw|?????????????njch????????????????????!6BOht?????uh[O6????????????????????????????????????????ABOS[]hjh\[[POHBAAAA#0<IQXUNI><;0#??????????????????????????????????????????????????????????????????????}????????????????}|}MOR[hrih[OMMMMMMMMMM)26BNOBA6)IO[htv??????th[QOGEI????????????????????#/<UagtuqdH</ ?#0<NWWULI<0)#???.0<CIJUXYWSI<0,)(().????????????????????05BGLIB>520000000000?#)0<DCCD<40
??GKQ\bn{??????{nUIAEG??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????DHMUVYaba^XWUTQHGCDD??????

?????????????????????????????????????????????????S[agmtttg[UTSSSSSSSS??????????????????????#/0131/-$#
??//37:<@DDCA</,//////)5<BNRNJCB=5)U[\agit????????ytg[U?s?m?g?_?g?s?????????????s?s?s?s?s?s?s?s?Ľ??????????????????????????Ľƽн׽н??????????????? ?????????????;?/?/?(?/?;?H?R?J?H?;?;?;?;?;?;?;?;?;?;čččĎĔĚĦĮĳĵĳĳĦĚčččččč????????????5?N?X?Z?b?`?N?5??ÓÎÇ?{ÇÓàëìïìàÓÓÓÓÓÓÓÓàÚÕÓÓÉ?|ÇÓàãìïùÿ??üùìàÓÒÏÐÓÓÔàáäàÚÓÓÓÓÓÓÓÓ?Z?Y?P?Z?\?f?k?s?????????????????s?f?Z?ܹѹչعع׹׹ڹ?????????????????g?N?5?(?????ؿٿ?????5???????????????g?Z?P?N?E?N?Z?`?g?s?????????????????s?g?Z?;?0?/?"???"?/?;?H?Q?T?^?T?H?B?;?;?;?;?????????????????????????	?????	?????h?]?[?O?B?@?B?O?[?h?j?n?h?h?h?h?h?h?h?h???????????????ʾ???;?O?L?;?.??	???ʾ????????¾¾??????????????????????????????m?`?\?T?V?`?i?m?y???????????????????y?mìâãìù????????????ùìììììììì??????s?o?m?s????????????????????????????????׾;ɾʾ׾????	??.?3?.?+?"??	???[?S?Q?V?Z?[?h?tĀ?t?t?tĀ?}?t?h?[?[?[?[????ĳĚčāāĎąěĦĿ?????
???
?????g?_?^?^?]?d?g?h?????????????????????s?g?ݿۿݿ????????????????ݿݿݿݿݿ?ŭŦŠŝŠťŭųŹž??ŹŭŭŭŭŭŭŭŭƚƁ?`?Q?O?\?oƁƎƧƳ????????????ƳƧƚ??????????ƻƻƽ???????????????????????ٿ	???޾ؾ׾Ҿ׾??????	?&?.?1?.?'?"???	?Ŀ????????????????????ѿտۿ޿????ݿѿľ???s?Z?>?9?B?M?Z?s????????????????????5?,?(?$?"?(?5?A?D?K?N?U?N?A?5?5?5?5?5?5?H?B?;?6?:?;?H?M?T?]?T?M?H?H?H?H?H?H?H?H?ο˿̿ݿ??????????? ?!?!???????ݿ???????øëäåìõ??????????????????H?<?5?4?8?F?H?U?a?n?zÇÓÕÞÓÐ?n?U?H?O?B?7?5?;?B?O?[?hāčėčā?{?}?t?h?[?O?/?(?)?-?/?<?B?G?@?<?/?/?/?/?/?/?/?/?/?/FFFFFF$F/F1F9F1F$FFFFFFFFFŠŞşŠŭŹŹŹŮŭŠŠŠŠŠŠŠŠŠŠ?H?C?=?;?1?/?/?,?/?;?H?T?Z?`?[?^?T?M?H?H???????y?m?b?_?i?y???????ѿ??????ҿѿĿ?????ŹŶŴŵŹ?????????????????????????????r?q?u?s?w?r??????????????????????????????"?.?7?;?@?@?;?.?"???????k?[?c?|?ûܻ??????????л??????x?k?l?f?a?l?v?x?????????????????x?o?l?l?l?l?!???????!?-?:?F?N?S?T?S?M?C?:?-?!?O?G?O?R?[?c?h?o?t?p?h?[?O?O?O?O?O?O?O?O?ֺ̺????????!?$?)?:?B?N?F?:?3?!????̻l?h?_?Z?T?_?l?v?x?y?x?x?l?l?l?l?l?l?l?l?Ľ½ĽŽνн۽ݽ????ݽнĽĽĽĽĽĽĽ?ŭťŠŖŔŇ?s?{ŇŔŠŭ????????????Źŭ?Ϲ˹ù????ùϹѹ۹ֹϹϹϹϹϹϹϹϹϹϹܹڹڹܹ޹?????????
???????ܹܹܹܼ@?;?9?9???@?I?M?Q?Y?f?s?s?q?n?f?c?Y?M?@?r?f?a?d?f?r?????????????????????????rE?E?E?E?E?E?E?E?E?E?FF$FAFHF=FE?E?E?Eٽ??????????????????Ľ׽ڽڽݽ????νĽ?????????????????(?4?A?O?Z?d?Z?M?A?4???;?0?2?;?>?H?T?T?Z?T?Q?H?;?;?;?;?;?;?;?;??????????????????????????????????????????????????????????????????????????????????s?g?Z?W?S?S?W?Z?g?s???????????????????Y?Y?o??????????!?*?+???????ʼ????r?Y????????????????????????????????????????????????????????????'?0?;?I?H?A?0??????H?H?@?<?;?<?A?H?U?U?a?h?l?m?i?b?a?U?H?H???ֺ??????????????ɺ????????????????ּ˼ʼƼɼʼּ?????????????????ּ?????(?5?N?Z?j?l?`?N?G?A?5?????????	?????????	????"?.?;?G?;?.?$?"??	?	?5?4?)?'?(?)?3?5?B?K?N?[?Y?Y?N?B?5?5?5?5?o?b?V?I?9?4?3?0?&?=?I?b?jǁǎǗǙǔǈ?o?y?s?l?d?d?l?y???????????????????????y?yÓÇÓÙàììíøõìàÓÓÓÓÓÓÓÓ????????????????????????????????????????D?D?D?D?D?D?D?D?D?EEEEE E.E*EEED?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D??n?j?^?X?U?Q?R?U?Y?a?n?zÇÔáÝÓÇ?z?n?H?>?<?/?%?#????!?#?/?<?@?F?H?J?L?O?H [ ? W < ) " Z h ~ B Z u Z C Z c j | % P J E @ e @ n A M C 0 = 1 K 8 4 Z $ 7 e D Q O 9 " 6  h y G N Z L > J O H \ X ! 5 D / D W L v @ 0 G 5 ( K i ! Y ? V U 9 ? ` I  Q  ?  f  m  ?  ?  n  ?  ?  `    Z  O  ?  1  {  y  ?  *  ?    ?    ?    ?  r  N  ?  ?  ?  ?  ?  P  ?  ?  ?  _  ?  h  (  ?    &  ?  ?  ?  ?  ?  ?  \  e  n  ?  ^  /  ?      ?  ?  ?  L  ?  9    ?  5  (  ?  S  !  ?  ?  ?  #  ?  b  ?  ?  ?  <<?9X<u;?o;?o??`B?????ě??u??`B?e`B??P?Y???o??C????
???
?49X??C??0 żě????ͽC???`B?t??\)??9X??j?}????w?+?#?
??????/?ě???\)?P?`???P?y?#?o?????`B????H?9?,1??'???C??H?9?t??L?ͽ?w??P?@??'aG?????aG??!???u?u?8Q??@??e`B??C??????]/?? Ž??㽡????9X??/?u??\)??Q콰 Ž?Q????????m?\??;d??S?B5]B,??A???BqA?JpB?A?iKBi?Bp_B?B??A?ܯB??A???B?B??B?{B5?%B5xB?SB?<B01?B??B%B+?B??B?BD?B??B?:B??B!?aBΠB??B
A?B??B?sB??B??BezB	5Bs?B)ԂB?tB?TB(cB??B??B%?	B}oB"LB!?AB!??B$?BP?B]B?pB ?B??B%qUB&C?B*?B??B%1mB(6?B,?1B
?pBf?B!/?B?B?DB?[B??B??B ?Bf?B	JBjxB\B??B??B	?B+?B-D?A???B9?A?b?B??A??LB |BIBa?B}A??gBA??\B??B?OB*aB5?B@kBN?B?{B07?B??B@ BVPB??B>?B@B?PB?B??B!??B??B?B
?\BO?B??BBvB??BDB:^B??B)L0B??B?B2B?BŮB&IEB?0B!?B!?kB!F?B??BN?BA?BdB!83B?rB%[?B&zQBGB?VB%dXB(??B,?!B
?B??B ?*B?nB?B?EB??B?9B@2BNB?pBh?B6_B?]B?B	??A??A#%/A?IKA??PA??A???A??ZA˯A?EgADY?$j?A?oA??LA?D?A?c?A?0^AV?5AM;?AnEA??RAG?AX?2Aۊ?A?[?A?Y?A?n?A??-B-?B?.AZqAw?RAC?A???A?n?A???AО(A?&?A??A???C??#A?>MA?{?At[A?r?A??{A_??@??%@?8W@t??Aڱ1@`?@?˯A)?2A?PH>????(b@??)@??C???A#B?A8?A??zA?8?A???A??u@?A???A?fmA???@BS5A1?A?$?A^,?A???Ba?A?GA˝?A?CC?P?C??dAȁdA??A?hA"?A???A?@A߈`A?yPAʍdÄ?Aʴ?ADl6?/?`A?c}A?`A?p
A???A?n?ASUAM??An?kA?jEAG??AX?A?~!AゅA?b?A?}?A???BkcB??AY?lAxπAB?A?N	A??aA???A?j+A?m?Aق?A?{?C??^A?\?A??mAwjHA?8A???A`??@?@?1?@t2A?g@]?Q@??A)E?A???>R>?? @?Oc@??VC???A"?;A7ϖA?RIA?h7A??BA??s@??SA??A?U?AŁv@???AI?A??ZA\?"A?xBP?A??Aˣ?A???C?G?C??EAȓ6A?~?               	                  !   0   	   	      	                      
               +            ?         1      5   '   
                  6      ,                              &      z                     9      #         "   4                     (                           !                  ;               1                     )            %            !            !   !                  '      -      -            !                        -   !                  7      #      #               !                                                         ;               /                     )                                    !                     '            '                                    '                     3      !                                          N?DOX Neu+NS)WN?(.O??N[|:O0?)M???O)*?O<??P|??O?N?w?N???NB?EP?cN??O#??N???N?vAO?N?g?O?U?OcN?NaENR?LO??iOBAO"tOH? O8??N??#N/m?O%kbO?	SO:?O^?Nb??NG?8M?4N?x?O???O?!O?tN?sO? KNAp?O?N|N?N??NC??NK^1O	?yN+#dNҾ?O# ?N?C?P	??Oc??O rN{	=N.??O26sOD/PO?N?O?+?N??O?1O.OC?kN?3?N?TO?[O ??N^??O
,'O4FgNh?O?LOq  ?     >  ?  ?  ?  ?  ?  ?  ?  ?  ?  }  u  ?  7  ?  ?  ?    &  f  ?  ?  ?  R  ?  
     ?  !  N  ?  ?  b  ?  ?    l  o  ?    ?  ?  ]  ?     ?  ?  9  ?  H  L  z    (  ?  ?  ?  v  ?    N  ?  ?  u  5  <  ?  P  K  	?  ?  r  j  ?  }  ;  	@  ?  ?   <ě?<?1<49X;?`B:?o%   %   ??o?o??o?t??ě??o?#?
?49X?D???e`B?u?ě???t???t???1??1???
???
???㼛???o??1??j??9X?49X??9X??9X?'?j?<j????j?ě????????????????h?@????C????+?o????o?o??P?C?????'t??m?h?0 Ž,1?,1?,1?0 Ž@??H?9?T???e`B?]/?e`B?aG???C??ixսq????%???
???罬1??{??1??vɽ?^5jnsz??{{zxnmjjjjjjjj???????	?????????mmovz{?????zmlmmmmmm2<=HQRSH<:2222222222_aimz?}zwmaaXU______??)5:@>@D>5)????kmrz??zumkjkkkkkkkk????????????????NO[htvtjh[QONNNNNNNN%)/13.	???KUanz?????????zn^RKK1/05Tamz????|yTH;/+1#(,/3452/#?????? ??????????????????????????????)/016<FGC=</))))))))?????????????????????????????????????????)/59:65)	???????????????????????359:BNVWWTQNJB>75.33"'*/6CHJHGDC@;6* ???????????????????????????????????????????(#????????!)+..1)%?????????????????????????????????????????????????????ntt??????????tojklnn????????

???????????????????????????357BNWUPNB>50/333333JNQ[ghig`[TNJJJJJJJJggmt??????????tmgggg?????
#/.*#
??????!#/<GHUXXVUH<:/'# !???????????????????? )6BLNB6)#          ??????????????????????????????????rt??????????????{trrhw|?????????????njch????????????????????)+3BOhuxyxtnh[OB;1+)????????????????????????????????????????ABOS[]hjh\[[POHBAAAA#04<IOVSLI<0,#???????????????????????????????????????????????????????????????????????????????????????????MOR[hrih[OMMMMMMMMMM).6BBKB<6)LO[hr???????th[TOHGL????????????????????#/<Uajlkf^H</&
#0<DINPFA<0#
?
,01<IQUVUPID<20/,++,????????????????????05BGLIB>520000000000?#)0<DCCD<40
??Uabn{??????{nbURJKOU??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????DHMUVYaba^XWUTQHGCDD??????

?????????????????????????????????????????????????S[agmtttg[UTSSSSSSSS??????????????????????
#/121/,#
??//37:<@DDCA</,//////')5BFB??;54)U[\agit????????ytg[U?s?m?g?_?g?s?????????????s?s?s?s?s?s?s?s?Ľ??????????????????????????Ľƽн׽н??????????????? ?????????????;?/?/?(?/?;?H?R?J?H?;?;?;?;?;?;?;?;?;?;čččĎĔĚĦĮĳĵĳĳĦĚčččččč???????????)?5?N?W?Y?a?_?N?5??ÓÎÇ?{ÇÓàëìïìàÓÓÓÓÓÓÓÓàÚÕÓÓÉ?|ÇÓàãìïùÿ??üùìàÓÒÏÐÓÓÔàáäàÚÓÓÓÓÓÓÓÓ?Z?Y?P?Z?\?f?k?s?????????????????s?f?Z?ܹعڹڹٹٹܹ???????????????????g?N?5?(?????ؿٿ?????5???????????????g?Z?P?N?E?N?Z?`?g?s?????????????????s?g?Z?;?6?/?"???"?/?;?H?O?T?V?T?H???;?;?;?;????????????????????????????	??	???????h?]?[?O?B?@?B?O?[?h?j?n?h?h?h?h?h?h?h?h?ʾ????????????ʾ???;?G?M?K?;?.??	???ʾ??????¾¾??????????????????????????????y?n?m?e?b?i?m?s?y???????????????????y?yìâãìù????????????ùìììììììì??????s?o?m?s????????????????????????????????޾׾Ѿξ׾??????	???"?(?"??	???[?V?T?[?f?h?h?t?|?z?t?h?[?[?[?[?[?[?[?[čĈĉĐċĚĦĿ???????
??
????ĿĳĚč?g?a?_?^?g?s?????????????????????????s?g?ݿۿݿ????????????????ݿݿݿݿݿ?ŭŦŠŝŠťŭųŹž??Źŭŭŭŭŭŭŭŭ?y?h?c?b?h?uƁƎƧƳ??????????ƳƧƚƎ?y????????????ƽƼƿ?????????????????????????????޾????????	???"?,?%?"??	???????Ŀ????????????????????ѿӿڿݿ????ݿѿľf?^?Z?P?M?K?R?Z?f?s???????????????s?f?5?,?(?$?"?(?5?A?D?K?N?U?N?A?5?5?5?5?5?5?H?B?;?6?:?;?H?M?T?]?T?M?H?H?H?H?H?H?H?H?????ݿؿ׿ݿݿ??????????????????????ùìéåêù??????????????????H?F?B?C?H?I?Q?U?a?n?o?y?{?z?w?n?h?a?U?H?O?B?:?9?>?B?O?[?h?tāā?}?x?s?q?h?c?[?O?/?(?)?-?/?<?B?G?@?<?/?/?/?/?/?/?/?/?/?/FFFFFF$F/F1F9F1F$FFFFFFFFFŠŞşŠŭŹŹŹŮŭŠŠŠŠŠŠŠŠŠŠ?H?C?=?;?1?/?/?,?/?;?H?T?Z?`?[?^?T?M?H?H???????y?m?b?_?i?y???????ѿ??????ҿѿĿ?????ŹŶŴŵŹ???????????????????????????????z?~???????????????????????????????????"?.?7?;?@?@?;?.?"?????????l?m?????ûܻ?? ???????ܻлû??????l?f?a?l?v?x?????????????????x?o?l?l?l?l?!???????!?-?:?F?K?O?I?F?@?:?2?-?!?O?G?O?R?[?c?h?o?t?p?h?[?O?O?O?O?O?O?O?O??????????????!?,?-?/?-?+?!?????????l?h?_?Z?T?_?l?v?x?y?x?x?l?l?l?l?l?l?l?l?Ľ½ĽŽνн۽ݽ????ݽнĽĽĽĽĽĽĽ?ŭūŠŝŖŘŠŭŹ??????????????ŻŹŭŭ?Ϲ˹ù????ùϹѹ۹ֹϹϹϹϹϹϹϹϹϹϹܹ۹۹ܹ߹???????????	??????ܹܹܹܼ@?=?<?;?@?L?M?T?Y?f?h?o?p?o?l?f?a?Y?M?@?r?f?a?d?f?r?????????????????????????rE?E?E?E?E?E?E?E?E?FFF$F9F>F4F$E?E?E?Eٽ????????????????????ĽʽϽϽнƽĽ?????????????(?4?A?J?M?U?Z?[?Z?M?A?4??;?0?2?;?>?H?T?T?Z?T?Q?H?;?;?;?;?;?;?;?;????????????????????????????????????????????????????????????????????????????????g?Z?Z?V?V?Z?Z?g?s???????????????????s?g?f?b?p??????????!?(?)???????ʼ????r?f???????????????????????????????????????????????????????????
??#?7???E?D?<?0?
???H?H?@?<?;?<?A?H?U?U?a?h?l?m?i?b?a?U?H?H?ɺ????????????ɺ????????????????ֺɼּ˼ʼƼɼʼּ?????????????????ּ????????(?5?A?N?Y?Z?a?V?N?A?5?(???	?????????	????"?.?;?G?;?.?$?"??	?	?5?4?)?'?(?)?3?5?B?K?N?[?Y?Y?N?B?5?5?5?5?o?b?I?E?;?7?8?=?I?V?b?h?{?ǌǖǗǔǈ?o?y?s?l?d?d?l?y???????????????????????y?yÓÇÓÙàììíøõìàÓÓÓÓÓÓÓÓ????????????????????????????????????????D?D?D?D?D?D?D?D?D?EE
EEE,E*EEEED?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D??n?k?a?_?[?W?a?h?n?zÇÑÓÞÛÓÇ?z?n?n?H?>?<?/?%?#????!?#?/?<?@?F?H?J?L?O?H [ ? W < )  Z h ~ B R u Z @ R c n | , P J E  a ; n A = A 7 =  K 8  R 0 9 e D Q O 9 " &  h y F N / L > - O A [ X   3 D / D W < s @ 9 G 1 ( G i ! P ? V U 5 ? O I  Q  ?  f  m  ?  ?  n  ?  ?  `  ?  Z  O  ?  ?  {  c  ?  i  ?    C  ?  Z  ?  ?  r  S  ?  5  ?  |  ?  P  ]  ?  K  ?  ?  h  (  ?    &  c  ?  9  ?  b  ?    e  n  &  ^  ?  ?    j  ?  s  ?  L  ?  ?  ?  ?  ?  (  |  S  ?  ?  ?  @  #  ?  b  ?  ?  1  <  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?}  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  o  f  ]  T     ?  ?  ?  ?  ?  ?  ?  |  h  W  G  8  %     ?   ?   ?   ?   v  >  5  ,       ?  ?  ?  ?  h  F  %    ?  ?  ?  ?  \  2    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  c  O  :  &    ?  ?  ?  ?  ?  ?  o  W  =  $  	  ?  ?  ?  ?  S    ?  F   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  s  N    ?  ?  e  0  ?  ?  e   o  ?  ?  ?  ?  ?  ?  ?  ?  i  O  3    ?  ?  ?  ?  v  K     ?  ?  ?  ?  s  \  >    ?  ?  	       ?  ?  w  :  ?  ?    }  ?  ?  ?  ?    ,  F  `  x  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?    ?  ?  ?  ?  ?  ?  ?  ?  }  j  X  D  2       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  l  6  ?  ?  l  '  ?  ?  f    ?  ?  -  T  P  ?  ?  ?  k  M  J  u  a  C    ?  ?  ?  ?  ?  Q  ?    	   ?  }  {  z  w  s  o  h  `  R  C  1      ?  ?  ?  ?  n  `  S  k  o  t  n  c  V  E  3  "    ?  ?  ?  ?  ?  ?  j  J  &    c  w  ?  ?  ?  ?  ?  ?  |  q  b  O  9    ?  ?  ?  ?  ?  ?  7  )        ?  ?  ?  ?    	  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  l  ^  _  @    ?  ?  ?  `    ?  ?  B    ?  ?  ?  ?  ?  ?  ?  ?  ~  w  q  m  i  e  a  ]  Y  U  P  L  8  x  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  J  ?  ?  2  ?  %  ?        ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  z  ?  ?  ?  &      	  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  i  H  W  _  d  e  a  Y  M  ?  -    ?  ?  ?  ?  N    ?  ?  *  f  _  W  }  ?  ?  ?  ?  ?  ?  p  _  L  3    ?  ?  ?  >  ?  ?  ?  ?  ?  ?  ?  ?  {  n  \  G  0    ?  ?  ?  ?    |  &  ?  ?  ?  ?  ?  ?  ?  ?  ?  v  a  B    ?  ?  ?  c  >    |  R  L  E  ?  8  2  ,  %            ?  ?  ?           ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  f  L  2       ?   ?   ?  ?  ?  ?  ?    
  	    ?  ?  ?  ?  >  ?  ?  S  ?  O  ?  ?            ?  ?  ?  ?  ?  ?  ?  u  Y  6    ?  ?  @  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  Y  )  ?  ?  a     ?    !           ?  ?  ?  ?  ?  h  &  ?  ?  ?  ?  l  5  ?  `  ?  ?    5  E  J  N  I  9    ?  ?  _  ?  _  ?  ?  ?   ?  ?  ?    x  r  j  b  Z  R  G  =  3  )           ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?      "  ?  J  ?  ?    3  K  [  `  J  $  ?  ?  M  ?  m  ?  ?  ?   ?  ?  ?  ?  ?  ?  c  .  ?  ?  p  M    ?  i    ?  F  ?  A   ?  ?    C  q  ?  ?  ?  ?  ?  ?  ?  ?  ?  Z  ?  ?  ?  7  g    ?  ?          ?  ?  ?  F  ?  ?  5  H  ?  ?  ?  G  {  ?  l  j  h  d  _  Y  S  J  A  3  "    ?  ?  ?  [  /    ?  ?  o  b  V  H  /  0  3  4  5  5  7  ;  B  J  P  S  U  X  Y  [  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  t  k  c  Z  R  I  A  8  0      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  j  :    ?  d    ?  ?  ?  ?  ?  ?  ?  p  Z  B  %     ?  ?  g    ?  u     ?  ?  ?  ?  ?  ?  ?  ?  {  ]  <    ?  ?  ?  ?  e  B  $  	  ?  ?  ?  ?    *  H  Z  W  D  #  ?  ?  p  !  ?  T  ?  Q  ?  ?  ?  ?  ?  ?  ?  ?  }  h  S  <  "    ?  ?  y  @    ?  ?  v          ?  ?  ?  U  M  8  
  ?  ?  ?  ?  ?  B  ?    *  ?    n  ]  L  :  !  	  ?  ?  ?  ?  ?  }  g  K  -    ?  ?  ?  ?  ?  ?  ?  p  [  B  '    ?  ?  ?  ?  l  D    ?  7  ?  9  *      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  p  ?  ?  ?  ?  ?  ?  ?  ?  ?  d  E  $     ?  ?  ?  ?  ?  ?  H  H  G  E  B  ?  ;  5  /  $      ?  ?  ?  ?  }  ?  ?  ?  L  K  K  J  H  ?  7  /  &            ?  ?           G  X  c  i  u  r  d  R  A  .    ?  ?  ?  ?  ?  +  ?  ^   ?                	    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?        (  &      ?  ?  ?  ?  |  Z  8    ?  ?  ?  m  [  ?  ?  ?  ?  ?  ?  ?  ?  ?  l  6  ?  ?  I  ?  N  ?  ?    ?    ?  ?  s  \  D  7  /  (  #        ?  ?  ?  ?  ?  M  ?  ?  ?  N  ?  ?  ?  ?  ]    ?  N  ?  :  ?  ?  ?  
?  ?  ?  ?  ?  l  t  v  u  u  v  r  n  g  `  W  A  #     ?  ?  a    ?  :  ?  ?  ?  ?  ?  ?  ?  ?  ?  }  a  8    ?  ?  X    ?  ?  H    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  j  \  O  A  N  E  =  5  ,  "        ?  ?  ?  ?  ?  ^  .  ?  ?  X    ?  ?  ?  ?  ?  z  n  \  F  *    ?  ?  t  J  "  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  t  V  0    ?  ?  c    ?  g  ?  C  W  o  O  (    ?  ?  ?  n  -  ?  ?  s  %  ?  I  ?  O  ?  |  5  ,  $      	    ?  ?  ?  ?  ?  ?  ?  ?  l  V  A  +        7  ;  0  %      ?  ?  ?  t  @  	  ?  z    z    ?  ?  ?  c  0  ?  ?  ?  i  ,  ?  ?  ?  e  .  ?  ?  G  ?  ?  ?  L  M  >  ,       ?  ?  ?  ]  (  ?  ?  ?  ?  `  3    
  ?  K  @  2      ?  ?  ?  ]  "  ?  ?  K  ?  ?  %  ?  ?  N  %  	?  	?  	?  	?  	?  	?  	?  	?  	y  	@  ?  ?  F  ?  X  ?  L  ?  U  ?  ?  ?  ?  }  n  `  R  D  7  *        ?  ?  ?  ?  ?  ?  ?  r  i  `  U  K  @  6  ,  !    ?  ?  ?  ?  C  ?  ?  >   ?   t    `  i  V  1    ?  ?  q  :  ?  ?  }  .  ?  ?  $  ?    $  ?  ?  ?  ~  r  f  Z  K  ;  *    
  ?  ?  ?  ?  ?  ?  ?  ?  }  d  K  1    ?  ?  ?  ?  ?  ?    k  X  C  /    >  w  ?  ;    ?  ?  ?  ?  ?  o  P  /    ?  ?  ?  ?  i  A    ?  ?  	7  	9  	6  	;  	  ?  ?  ?  U    ?  ?  ?  ?  9  ?  9  ?    P  ?  ?  ?  ?  s  S  1    ?  ?  ?  l  ?  ?  |  ?  |    ?  >  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  h  7  ?  ?  8  ?  >  ?  I     ?  ?  ?  ?  ?  ?  }  G  
  ?  ?  =  ?  ?  A  ?  b  ?  f