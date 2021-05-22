CDF       
      obs    Q   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?����+     D  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�&�   max       P���     D  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��G�   max       <e`B     D   4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>z�G�{   max       @E���R     �  !x   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vv�Q�     �  .    effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q�           �  :�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @�C�         D  ;l   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �1'   max       ;�`B     D  <�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B-�e     D  =�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�[   max       B-�O     D  ?8   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�[?   max       C�vi     D  @|   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��3   max       C���     D  A�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ]     D  C   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =     D  DH   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /     D  E�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�&�   max       P'�     D  F�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�f�A��   max       ?�1���-�     D  H   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��S�   max       <49X     D  IX   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�=p��
   max       @E���R     �  J�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vv�\(��     �  WD   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q�           �  c�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @��          D  d�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�     D  e�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�kP��{�   max       ?�1���-�     P  g   1         &            ]                        3            !                     (                                    
                     %            <         &   "   .       *          	   $   
      "                        
   	                        Pfv�N�͜M�&�O���O?Z�N%9~N��9P���NM��O��YO���O�>#N�r�O�nTNsaP$aOD��N���N�3O��O��~Nnj�O���N���Nǫ�O!�@PdO��NYN�՚Nۃ�N��N# �Ob^XN�ăN��NZ��N�`NA�[N��O�O~��N�p�N��N���O��O���N��O�� P~J�N���Omq�O!`P�N���Oj�Ob6�O��N/-�N�}OWR�N�j�OQ�O�N�w�Nb=�O+kOIe'O��O���N�NK��N�xOX(�N�d�NQ
AO��O���N��kN��{O$ Z<e`B<49X<#�
;��
��o��o�o�o�D���D�����
��`B�o�o�#�
�49X�D���D���D���T����C���t���t���t���t����㼛�㼛�㼛�㼬1��1��1��9X��j�ě����ͼ���������/��`B��h��h��h���o�o�o�C��\)��P��w��w��w�49X�H�9�H�9�L�ͽL�ͽL�ͽ]/�]/�e`B�ixսm�h�u��%�����\)��t���t���t����w���w���w���w���������ȴ9�ȴ9��G�HWn�����������zUGMLHcmz������~ztmjeeggcc}�������~}}}}}}}}}}}��������������������157<BEN[hnkf_[NB52*1�����������������������������������k������������{nick��������������������!/<H[anmYH:4/# ��������������������6<HUaglnrnl`UH;32326DHUagnpqpnhaUPHGHIHD����

���������������������������)6Ohtr{�}ti[B6-mz������������zynihm��������������������<<>HUaaga]WUQHH<<<<<Uaz�����unpna`UNKKOU��������������������QTaemmmjfa^TROQQQQQQdmx{pz�����������tdd�����������������������������������		������Uaz������������zaTOU�����������������{|����������~����������-/47;CGHJKLMJJJH;/)-V[gtz|ytqg[SPOVVVVVV"#/:<GDBD<8/$#��������������������!$,35BENW[]b`[N5)����������������������������������������������������������������������������������������������zz�����}zyzzzzzzzzzz������

����������)6DN6/*%���*6BOS[XVQOIB651*****:<BHTPKH=<;:::::::::����������������������������������������BOVbjst������th[OIBB










��������������������[��������������t[SS[���� ������!)5BHN[gt����tgQB5#!#/<FHLLIH=</#��	#0<Un{}yq[0#
�����	

�������������������������������|uw�����������������������������������������/0<FFFCD<630-+//////0<IKLRUVUTPI<0(%#$&0u{������������}{xsru����������������������������������������;<==IU^`aaUMI<;;;;;;./7<HOTIH</.........GOP[hpomorph][ONGFDG����������tqpqstx�����������������������������������������������������������]gntwytrgfd\]]]]]]]]pty��������xtroppppqz���������������znq���������������������������������������������������?BN[ot���{tg[NB8<<@?]anqsxz{�|zyrnha^YX]��������������������!#)/GC@=6443/#�s�g�[�^�g�����������������������������s�����������������������������������U�P�T�U�a�e�n�q�n�a�U�U�U�U�U�U�U�U�U�U������������������
��#�-�8�<�?�H�<�/��������������������������ĿɿԿӿҿѿĿ��U�T�Q�R�U�]�a�d�n�p�o�n�j�a�]�W�U�U�U�U���������������������������������������������m�S�M�T�m�������Ŀ���#�1�/����ѿ��$�#��$�(�0�;�=�>�B�=�;�1�0�$�$�$�$�$�$�H�;�6�4�4�;�B�H�T�m�x�������������z�a�H�����������z�a�]�l�q�z������������������������������������������
������������ùöþ��������������������������������ù�f�L�@�>�6�5�@�M�Y�r����������������r�f��Ʒƿ�����������������������������������T�.�#� �&�6�;�T�`�m�y���������������m�T���������������������$�'�$�$������4�1�)�(�"�#�(�0�4�9�A�J�J�A�>�A�C�A�4�4�#�!������#�*�/�<�@�<�/�/�#�#�#�#�#�k�^�R�Z�f��������ʾ׾���پʾ�����k�I�>�I�V�b�n�{ŇŔŠŬŪŤŠřŋ�{�n�U�IčĉćčĚĚĚĦĳĳĳĬĦĚčččččč������ĿĳįĲĳ�������
�#�4�.���
��������������������� �)�,�)�(�������������������������������ɾƾ����������	���������������	���"�'�/�3�2�/�"��	���տÿ��������ĿͿ���%�,�,�4�C�?�5��	�������������������	��6�H�R�R�H�"��	�6�0�*�)�*�6�C�D�M�C�6�6�6�6�6�6�6�6�6�6���������������������������������������������������������������������������������a�^�U�T�N�U�[�a�n�zÇÉÇÃ�z�y�t�n�a�aǈ��}ǈǔǙǡǥǡǔǈǈǈǈǈǈǈǈǈǈ�����������������������������������������z�u�s�z�ÇÈÓÚÞßÞÓÇ�z�z�z�z�z�z������������������������s�k�g�b�g�j�s�|���������s�s�s�s�s�s�s�s����������������������������������������ā�w�uāčĚġĚĕčāāāāāāāāāā�u�t�n�uƁƎƔƎƆƁ�u�u�u�u�u�u�u�u�u�u�������������������ĿɿѿҿֿٿٿѿĿ����лû����������������ͻ��������ܻм������'�4�@�A�@�>�5�4�'������B�>�6�3�6�B�O�Q�[�\�[�O�B�B�B�B�B�B�B�B�����������$�.�(�$�����������������ͺºźں������-�F�P�K�-�$��������������������������ĽƽȽӽݽݽ׽нĽ����������ûͻлܻ�ܻлû������������������پԾѾѾԾ׾���	���$� ���	�����A�?�G�a�d�S�[�u¦¯¾��������¿¦����׾Ծ׾�����	������	���������������������	�������	���$��������$�0�=�>�I�J�I�C�=�4�0�$���������t�i�c�]�c�s��������������������E*EEEEEEEE*E7ECEIEPEPEPELECE7E*E*�����~�z������������ʼӼѼʼļ������������}�|�{�w�x�|�����������Ȼ��������������ֺɺ����������������ɺֺ�������ֻx�n�l�c�i�l�x�~�}�|�x�x�x�x�x�x�x�x�x�x�������������ĽнؽнĽ��������������������۽�������(�4�A�I�D�4�(����������%�'�4�@�M�Y�a�Z�Y�M�I�@�4�'���������������������
������
������3�-�2�;�B�E�L�V�e�r�~���������~�r�Y�@�3�S�G�F�:�1�0�:�F�S�_�c�_�Y�U�S�S�S�S�S�SD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�Dƹù��������ùϹܹ������������ܹϹ������������	���&�-�#��	������������������ŹŷŸŹ���������������������������Ƽ����������������ʼ��������׼ʼ������������
�����������������������ā�~āąčĚĦĦĪĦĚčāāāāāāāāĿĶĿ��������������������������ĿĿĿĿ�Ŀ��������ſѿڿ������������ٿѿ���	����#�(�*�5�<�5�1�(���������������������������������������E�9�?�T�`�o�y�������Ľݽ�ܽн������`�E�n�e�V�dÇÌÓàù����������ûùìÓ�z�n���������ùϹ׹ܹ����������ܹϹù��3�0�'�"�'�3�=�@�L�Y�d�a�Y�L�@�4�3�3�3�3E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFFF E� - \ N 4 > � ! 8 b D 8   S A b # M [ D ` Y 9 H 4 ^ F e ` g � " 8 D 0 H W ) L @ H H b 3 k : U L Y ' E = |  h C 6 > J ~ \ A S  Y ) F E ; - @ h H g c K x e E y r �    �  1    b  �  �  �  �  �  �  �  �  �  <  B  �  �    �    T  �  m    �  l  |  �    |  �    /  �  �  ;  _  �  d  1  F  C  "  C  �  �  G  =  8      S  ^  i    �  �  u  �  �  �    �  �  �  w  w  �  (  v  e  }  �  0  �  �  j  �  >  �  ��\)%   ;�`B�o��o�ě��t�����o��1��/��h�u�\)�T���y�#��h���
��t��8Q�C���j�8Q�'ě��o�q���<j��1��`B�o�49X��/�'t����+�+�t�����w�L�ͽ<j�t��H�9��C��0 Ž�P�]/�ě��T���m�h���㽟�w������
��j��e`B��%��E���+��-��^5��+���P�\�\�������-��-�� Ž�"ѽ�^5��Q��;d���`��G���S��1'B�A��B�BPB;=BoB��B*�B�B�
B&BlB�B#��B��BE_B |CB��B��B~�B��A�3B�B�ZB"��A���B �BB ��A�ˏB	;�B
UB�B�tBYB��B7tB�{Bz�B *jB�B�B7�B/�B�,BZ�B!�B$b�BxB5�BB�B	.�BLCB&!BޣBS�BɵB".JB"4�B&-�B&@�B)E�B��B!XB&��B�BYBOqBmBB+�B-�eB	H�B
)�B��B��BѳBߑB�B�{BP�BHZBCA���B&B1FBl�B~'B��B*��B�B��BBO�BpB#�@B��B?sB @YB6mB��BA�Bf�A��B@BB��B#8�A�[B @XB��B ��A�~aB	A@B�(B�B�IB[�B��B<0B��B��B <�B��B��B@?B;�B�pBA By�B$@�BC_B
>�B@B	=pBMB&�JB�uB�B�mB!��B"e/B&@B&?�B)C�B��B!IB'`B�^B>�B��BY�B*�cB-�OB	H�B
A?B�\BHXB�B?�B	�B��B@�B�A�C�AѺ�A�doA��dAu�NA�)A��Ay�B
:�A���A��yA��?AϬ�@�3B;3AjtB�A9P�A��AI"A�4A߾�A���A�p�AKjlA��A��!A��SB o�A�)A��A�g�B7A���A��@���A�īA�{�A�'MB-�Ax@�`*@��A�-B��@T7A$��@��{AXA�A���AXuxAY\�B
A�?oC���@�N|@��@B�@�F7A$�|A4�R@���A��[?�2Q@��C�	6>�+�A�iA���@��A'yA���A�f�A},A��$A��(A��Aʿn>�[??��C�viA�n+A�#ZAƅ�A��1Au��Aŋ�A��MAy�	B
>kA��A�mA�=�Aπ�@��BRXAkB�DA9\A�1�AI �A�~�A�A�oPA�hAI�A�=�A�~�A���B >�A���A�k�AƊ�B?�A��ZA�y�@��IA���A�_�Aݗ�B�]Ax��@��@ɪ�A�p�B��@T`�A"jU@�� AW�A���AV��AY�B
#�A�C��;@��@�@D@�x�A#�A5��@�YAA�:?��@�'�C�$>��3A���A��R@�	�A�Aހ�A��/A|�?A��A�~�AO;A̿>�F?�o�C���   2         &            ]                        3      	      !                     )                                    
                     %            <         &   #   .       +   !      
   $         "                        
   	                           3                     =      !   !               +            #         )            7   '                                          #            #   !         7            /            #                  !                  !                     '               #                     /                        #                     %               #                                          #                        +            /                              !                                       '            P
s�N�͜M�&�OJֹO,�N	�dN�G�P'�NM��O���O��O	?NA�OmJ�NsaO�UAO$uVN���N�3O"}8O��~Nnj�O��N�Nǫ�N�3OO�saO���NYN�՚N�
6N*�(N# �Ob^XN�ăN��N4��Nm�*NA�[N��O�O~��N�p�N��NP��OT�OmVN��O�� P$�MN���OO!`P�N���Oj�O*�Oyr�N/-�N�}O/�N�j�O&�HO�N�w�Nb=�O+kOIe'O��O��N�NK��N�xO>AN�d�NQ
AO��O���N��kN��{O0  �  h  �  �  �  "  �    8  �  Z  "  �  d  �  �  �  �  �    "  g  7  4  J  /  �    �  �       _  �  M  *    g  E  �  L  =  ;  �      $  �  �  �  K  H  	�  S  �  E  �  p      :    \  q  ~  c  �  D  �  R  ~    �    <  ,    I  �  �  �%   <49X<#�
���
�D���o�D���t��D����o��`B�u�#�
�#�
�#�
���ͼe`B�D���D��������C���t����
���
��t���j�t���9X���㼬1��9X�o��9X��j�ě����ͼ�/��/��/��`B��h��h��h���\)�#�
�+�C��\)�H�9��w�0 Ž�w�49X�H�9�H�9�ixս]/�L�ͽ]/�m�h�e`B�y�#�m�h�u��%�����\)��t������t����w���w���
���w���������ȴ9�ȴ9��S�bnz����������zn_XVYbcmz������~ztmjeeggcc}�������~}}}}}}}}}}}��������������������/569>BN[^gjid][NIB5/������������������������	������������|���������������}wv|��������������������#/<HYajlaVH8/#��������������������9<<HU[aaba``UH><9899SUaennnnaURKSSSSSSSS���

��������������������������$)6BOV[lme^OB6)#jmsz�����������}zqkj��������������������<<>HUaaga]WUQHH<<<<<TUZanz��}zsnlfaUQQRT��������������������QTaemmmjfa^TROQQQQQQjm|�������������yhj��������������������������������������		����������js������������zmhcej�����������������~���������~����������-/47;CGHJKLMJJJH;/)-X[gty{xtmg][RQXXXXXX#,/<<<</-#��������������������!$,35BENW[]b`[N5)����������������������������������������������������������������������������������������������zz�����}zyzzzzzzzzzz������

����������)6DN6/*%���*6BOS[XVQOIB651*****:<BHTPKH=<;:::::::::����������������������������������������JOPWdkt�������th[OKJ










��������������������Xat�������������tb[X���� ������ENP[gt�����tsg^[NCEE#/<FHLLIH=</#��	#0<Un{}yq[0#
�����	

���������������|�����������������z|����������������������������������������/0<FFFCD<630-+//////.0<IJOSSSNI<10*'%&).u{������������}{xsru����������������������������������������;<==IU^`aaUMI<;;;;;;./7<HOTIH</.........GOP[hpomorph][ONGFDG����������tqpqstx�����������������������������������������������������������]gntwytrgfd\]]]]]]]]pty��������xtroppppx����������������zrx���������������������������������������������������?BN[ot���{tg[NB8<<@?]anqsxz{�|zyrnha^YX]��������������������!#*/<B?<6342/#�s�j�i�l�s�����������������������������s�����������������������������������U�P�T�U�a�e�n�q�n�a�U�U�U�U�U�U�U�U�U�U�����������������
��#�$�.�5�2�/�#��
���Ŀ������������������������ĿƿѿѿпͿ��U�T�R�R�U�]�a�d�n�o�n�n�m�a�\�V�U�U�U�U���������������������������������������������x�v�{���������Ŀҿ���������ѿ��$�#��$�(�0�;�=�>�B�=�;�1�0�$�$�$�$�$�$�T�H�;�7�5�5�;�D�H�T�a�m�v���������z�a�T�z�p�a�m�s�z���������������������������z�������������������������� ���������������������������������������������������żf�N�B�?�>�M�Y�f�r������������������r�f��Ʒƿ�����������������������������������T�;�3�-�.�6�;�F�`�m�������������y�m�`�T��������������� �����"�$�&�$�"�����4�1�)�(�"�#�(�0�4�9�A�J�J�A�>�A�C�A�4�4�#�!������#�*�/�<�@�<�/�/�#�#�#�#�#������u�p�s����������������Ǿþ��������I�>�I�V�b�n�{ŇŔŠŬŪŤŠřŋ�{�n�U�IčĉćčĚĚĚĦĳĳĳĬĦĚčččččč������ĿĶĳ���������
�#�'�/�-���
��������������������)�*�)�&���������������������������������ɾƾ����������	����������	���"�,�%�"��	�	�	�	�	�	����ݿοɿʿݿ��������%� �������	������������������2�D�N�O�N�H�/�"��6�0�*�)�*�6�C�D�M�C�6�6�6�6�6�6�6�6�6�6���������������������������������������������������������������������������������a�Z�a�d�m�n�z�{�|�z�s�n�a�a�a�a�a�a�a�aǈ��}ǈǔǙǡǥǡǔǈǈǈǈǈǈǈǈǈǈ�����������������������������������������z�u�s�z�ÇÈÓÚÞßÞÓÇ�z�z�z�z�z�z������������������������s�m�g�d�g�n�s�x���������s�s�s�s�s�s�s�s����������������������������������������ā�w�uāčĚġĚĕčāāāāāāāāāā�u�t�n�uƁƎƔƎƆƁ�u�u�u�u�u�u�u�u�u�u�������������������ĿɿѿҿֿٿٿѿĿ����лû����������������ͻ��������ܻм������'�4�@�A�@�>�5�4�'������B�>�6�3�6�B�O�Q�[�\�[�O�B�B�B�B�B�B�B�B�����������$�&�$���������������������׺ֺϺӺֺ�������%�-�!�������Ľ����������������������ĽŽѽ۽ܽսнĻ��������ûͻлܻ�ܻлû������������������پԾѾѾԾ׾���	���$� ���	����¦�n�b�^�n�t�r­³������������¿¦����׾Ծ׾�����	������	����������������	��������	�����$��������$�0�=�>�I�J�I�C�=�4�0�$���������t�i�c�]�c�s��������������������E*EEEEEEEE*E7ECEIEPEPEPELECE7E*E*�����~�z������������ʼӼѼʼļ����������������������������������û�������������ֺɺ������������ɺֺ�����
�����x�n�l�c�i�l�x�~�}�|�x�x�x�x�x�x�x�x�x�x�������������ĽнؽнĽ�����������������������������(�4�>�G�A�?�4�(�����������%�'�4�@�M�Y�a�Z�Y�M�I�@�4�'��������������������
������
�������غ3�-�2�;�B�E�L�V�e�r�~���������~�r�Y�@�3�S�G�F�:�1�0�:�F�S�_�c�_�Y�U�S�S�S�S�S�SD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�Dƹù��������ùϹܹ������������ܹϹ������������	���&�-�#��	������������������ŹŷŸŹ���������������������������Ƽ��������������������ʼּ�����ּʼ������������
�����������������������ā�~āąčĚĦĦĪĦĚčāāāāāāāāĿĶĿ��������������������������ĿĿĿĿ�Ŀ������ǿѿݿ������������ݿֿѿ���	����#�(�*�5�<�5�1�(���������������������������������������E�9�?�T�`�o�y�������Ľݽ�ܽн������`�E�n�e�V�dÇÌÓàù����������ûùìÓ�z�n���������ùϹ׹ܹ����������ܹϹù��3�0�'�"�'�3�=�@�L�Y�d�a�Y�L�@�4�3�3�3�3E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFFE�E� 1 \ N 2 7 � # : b ? 5 / B < b # J [ D Z Y 9 H 1 ^ H P R g �  c D 0 H W & T @ H H b 3 k L ' F Y ' D = M  h C 6 . A ~ \ 6 S $ Y ) F E ; - 2 h H g \ K x e E y r ~    x  1    �  \  z  �    �  ~  l    a  �  B  �  �    �  `  T  �    �  �  �  �      |  �  r  /  �  �  ;  C  �  d  1  F  C  "  C  n  �    =  8  �    Z  ^  i    �  r    �  �  z    a  �  �  w  w  �  (  !  e  }  �  �  �  �  j  �  >  �  �  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  �  �  E  �  �  �  �  �  �  �  �  �  x  =  �  k  �  ^  �  �  h  _  V  I  :  *      �  �  �  �  h    �  �  C    �  v  �  �  {  r  j  a  Y  O  E  :  0  &      	     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ]  -  �  �  @  �  M  �  �  �  �  �  �  �  �  �  �  �  �  �  }  i  N  *  �  �  �  U            $  /  ;  F  1  �  �  a  K  V  `  k  v  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  n  ^  P  B  7  /  &  �    R  �  �  �  �    �  �  �  4  �  z  %  �  *  n  F  ,  8  8  8  7  6  .  '      �  �  �  �  �  |  d  K  1     �  �  �  �  �  }  j  R  6      �  �  �  �  �  m  L  )    �  P  W  Y  V  N  C  8  +       �  �  �    V  %  �  �  �  �  �  �  �  �         "      	  �  �  �  �  c  I    �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  W  6    �  �  �  �  Z  c  ]  F  .              �  �  �  �  w  G  %    �  �  �  �  �  �  �  �  �  v  g  X  J  ;  )     �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  t  E    �  �  u  :  �  R  �  �  �  �  �  �  �  �  n  P  /  
  �  �  �  ]  &  �  �  ,  �  �  �  �  �  �  �  �  �  v  e  V  H  9  +        �  �  �  �  �  �  �  �  �  �  �    y  n  ^  N  9    �  �  �  �  o  �  �  �  �  �  �        
  �  �  �  ?  �  b  �  k  (  �  "    �  �  �  �  �  �  �  l  P  >  `  b  I  '     �  z  A  g  f  d  c  `  [  U  O  F  8  *    
  �  �  �  �  �  d  A  �  '  2  %    �  �  �  �  �  ]    	  �  �  �  �  �  �  �    .  4  0  %      �  �  �  �  ^    �  �  #  �  L  �  f  J  ?  5  *        �  �  �  �  �  �  y  m  a  T  ?  *        !  +  ,  -  .  .  *  !    �  �  �  �  P    �  �  2  �  �    I  o  o  |  �  �  �  q  M    �  �  _  �  {  �    �  �    �  �  �  �  �  b  B  #    �  �  �  �  �  �  �  �  �  �  �  �    u  j  `  V  K  <  '     �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  w  [  @  +    �  �  �  �  �  x  ~  ~  |  q  e  X  I  ;  )      �  �  �  i  =    �  n    �  �  #  ?  X  w  �  �  �     �  �  �  v  F      A     �  _  X  R  K  B  /    	  �  �  �  �  �  �  m  W  @  *     �  �  �  |  k  Z  J  7  "    �  �  �  �  �  �  v  J  �  �  \  M  9  &    �  �  �  �  �  �  h  G    �  �  _  .    �  �  *         �  �  �  �  �  �  �  z  \  <    �  �  �  �  t              	    �  �  �  �  �  V    �  �  9   �   �  _  b  e  d  Z  O  D  9  .  "        �  �  �  �  �  �  �  E  :  /  $        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  v  s  r  r  q  p  o  m  i  d  _  Z  U  Q  L  9  (  %  !      	  �  �  �  �  �  �  e  F    �  �  v  =    �  �  �  z  �  �  �  �  �  �  �  �  �  �  l  '  �  �  ;  1  &          �  �  �  �  �  �  q  M  (    �  �  k  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  a  M  :  �  �  �    C  �  �  6  �  �  �    ?  `  }  �  �  \  �  z  H  �  �  �  
    �  �  �  �  h  0  �  �  S  �  �  �  7        #        �  �  �  �  �  �  x  T  .  �  �  n  7    �  �  �  �  �  �  �  �  �  �  �  }  w  r  o  m  j  g  d  b  �  �  �  �  �  �  �  �  }  d  D    �  �  �  >  �  �  g  L  @  y  �  �  �  {  F    '  Z  Q  =    �  �  >  �    "  j  K  B  5  $    �  �  �  �  �  �  X  !  �  �  2  �  w     �  �  �    9  H  G  C  :  ,    �  �  �    D    �  4  �   �  	�  	�  	�  	_  	A  	  �  �  `  $  �  �  2  �  a  �  b  �    w  S    �  �  �  �  e  -    �  �  �  �  E  �  �  l    �   �  �  p  N  +    �  �  r  7  �  �    
o  	�  �     )  C  T  X  E  4  "    �  �  �  �  s  =    �  �  K     �  +  �  h  !  �  �  �  �  �  �  �  �  �  Y    �  {  $  �  K  �  q  -  �    J  j  i  W  ;    �  �  m  1  �  �  �  -  �    �  <  �      
           �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  y  [  ;    �  �  �  �  �  �  �  u  \  C  %  /  :  4  "    �  �  �  o  A    �  �  7  �  !  V  �  �        �  �  �  �  �  �  �  w  [  <    �  �  �  e  (   �    >  X  X  K  <  (    �  �  �  }  .  �  w    �  3  �  k  q  `  K  @  "  �  �  �  M    �  �  \    �  K  �  t  �  �  ~  p  c  V  K  A  7  -  "        �  �  �  �  �  �  �  �  c  a  ]  W  H  1    �  �  �  t  A  
  �  �  ^  #  �  �  k  �  �  �  �  �  �  �  �  m  F      H  "  �  k  �  ~  �  y  D    �  �  �  �  _  1     �  �  l  :    �  �  J    :  d  �  �  �  �  �  z  [  ;    �  �  �  �  ]  <    �  �  �  �    Q  @    �  �  }  A    �  �  �  f    �  �  �  M     �  ~  p  b  S  D  0      �  �  �  �  u  H    �  �  s  4   �    �  �  �  �  �  �  �  o  V  :  "    �  �  �  l  <    �  �  �  �  �  �  �  �  �  �  �  o  D    �  �  }  E    �  t  �  	    �  �  �  �  V  "  �  �  i  *    �  �  K  �  �  *  <      �  �  �  �  c  H  .    �  �  �  `    �  a  �  �  ,  d  �  u  Z  7    �  �  �  n  @    �  �  y  D    �  �    �  �  �  p  J  '      �  �  �  �  K    �  �  �  �  �  I  5    �  �  �  �  �  �  �  �  �  �  �  �  �  �  V    j  �  e  D  $    �  �  �  �  k  X  O  &  �  �  �  y  `  Q  H  �  �  �  �  j  D    �  �  �  S  &  �  �  �  p  7  �  �  E  �  �  S    �  L  �  U  �  �  E  �  �    �    �  �  c  �