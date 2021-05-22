CDF       
      obs    K   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��+J     ,  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M� P   max       P�J�     ,  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �+   max       ;�`B     ,      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��\)   max       @F��G�{     �  !0   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v~�G�{     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P            �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���         ,  98   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �)��   max       �ě�     ,  :d   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       B =�   max       B4�o     ,  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B4��     ,  <�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >U��   max       C��`     ,  =�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >M��   max       C���     ,  ?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          b     ,  @@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E     ,  Al   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?     ,  B�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M� P   max       P�?h     ,  C�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���D��   max       ?�Ov_خ     ,  D�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �1'   max       %        ,  F   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���
=q   max       @F��\)     �  GH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v~�G�{     �  S    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P            �  ^�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���         ,  _P   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?<   max         ?<     ,  `|   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�u%F
�   max       ?�	� ѷ     p  a�   +                  	      	            0      )         A   -            b            "                     
         /                     
         	                                              
   %         	               *      !   	   "O(�N��O�1�N�#O�%N0�EN���N�OhN�ɂOQ[�Nl�N��P��N,PP
��O:� N#;P�:�PSS�ObFN�͆O���P�J�N���N���O-H O�>aO��N�˘N�aTO�0ONx%N8N��O�ĎO�A�O�2�O�N`/4N���N:SNE�NP=jN��|N�0O,��N�*XO���OI�N6��O�^O�-O�OR�~O)*�N�G�N�E�O��N8'�O��%O&N�$�P ��N�P
�N#��M� POlW�N�R�O	��O���OH��N��CNl		N��T;�`B%   �o�o�D���D����o��o��o���
���
�ě��ě��ě���`B��`B��`B��`B��`B�t��49X�49X�49X�e`B�e`B�e`B��o��o��t����
���
���
��1��1��1��9X�ě����ͼ��ͼ�����/�o�o�+�+�+�+�\)�\)�t�������w��w��w��w�#�
�#�
�0 Ž0 Ž49X�<j�<j�@��D���H�9�L�ͽ]/�]/�q���}󶽟�w���+����������������������������������������69:B[dhkz�th[IB3+-68BCOS[`hkhee[PONB:88UG<75534<IUbpxurkebU��������������������(��*56?BHN[``^[XNDB@5**����

����������
#1FKJB<6/
����"#/;<C</#"""""""""""

����� 
_iz���������wg_����������������������������������� �����������������������������#@{�����tUI<���|����� 	�������{wv|����������������������th[STZ[^htz�������"%*./27<I`jjeZUN?</"anz��������������n[a&)5;BCKNONLB:52-)'&&��������������������{�����������������~{18BN[u����|ywsg[N;21BIUbmy}~��zgbUIEEA@B����������������������������������������")06BCOQSYOB6)$*3COSUUKEC6*�������������������������)5BR[gtod[NB( ��������������������_bmz���������vma`^^_���������������z}����������������������������������������}��������w}}}}}}}}}}������������������������������������������������������������������������������\agnz�������zsngaZY\��������������������otv�������������yrno����������������������������������������!)/6BO[htusnh[6	����
0<OW[UQ<0����P[`gt~�������tg\[OPP��������������������������

�������#$02<>C@<80-%# ��
####
��������������������������������������������������)O[h��th[B)����������������||����t{}������������{vttt������
������������������������������x�����������������~x$)6BKBB6)'$$$$$$$$$$MOX[hijh[OMMMMMMMMMM����������������������������������������358BNXYSPNONDB543223dgjt����������~vojgd�����������������
 #&*&#
 ����'/<@<<<82/)%(+''''''��������������û��������������ûлܻ�����ܻػл��/�-�'�/�<�H�P�U�_�]�U�U�H�<�/�/�/�/�/�/���y�m�c�a�m�q�y�����ĿϿѿοĿ����������.�"�"� �"�%�"��"�*�.�;�E�G�I�G�D�;�.�.�!�������ʼ����������ʼּ�����!�����������������������������������~�}�����������������������������������{�s�o�m�s���������������������ùõìëèéìù����������������ùùùù������"�)�6�B�O�[�`�[�R�B�9�2�)���H�H�F�H�I�U�W�^�X�U�H�H�H�H�H�H�H�H�H�H�˾׾۾���׾׾ʾ��������������˾˾˾�ƳƚƉ�|ƁƎƳ���������$�B�?�?�;����Ƴ���	���	����"�#�"�����������y�u�T�H�T�`�m�������Ŀѿ˿˿ſ�������¦¢¦²¿��������������������¿²¦�M�L�A�M�Z�f�n�f�Z�Z�M�M�M�M�M�M�M�M�M�M�������u�d�N�)�9�g������������� �������׿���b�b�j���������Ŀѿֿۿ��
�����Ŀ��ʾ������������������������ʾ׾���۾��a�a�Z�^�Z�T�H�;�3�/�"�$�/�6�;�H�T�a�a�a�	�����׾ʾ��������ʾ׾����	����	���������f�Q�`������!�F�T�^�R�!���޺ɺ��s�r�n�s�~���������������������������s�s�4�+�(�4�@�A�A�M�Z�\�f�h�n�f�a�Z�M�A�4�4�����������������������ʾξѾо˾ʾ������5�)�����������������5�B�P�X�W�N�B�5�!����!�-�:�F�S�_�l�x����|�_�F�:�-�!�5�,�+�(�'�(�5�:�A�N�Z�b�Z�N�L�C�A�7�5�5ǈǄ�{�o�b�V�R�V�b�f�o�{ǈǔǖǘǔǉǈǈ�S�P�S�V�S�Q�R�S�_�l�x�������y�r�q�l�_�S��޾׾Ӿʾ¾þʾ׾���	�����	�����ŠşŠũŭŵŹ��������ŹűŭŠŠŠŠŠŠ�������׾Ӿ׾�����	�����	�	�����������������������*�C�M�T�V�C�6�*�������������*�6�C�P�U�X�R�O�M�C�#��5�(�����������(�5�A�N�S�Z�^�g�g�N�A�5�{�z�n�j�b�U�b�n�p�{ŇŔŜŝŘŘŔŇ�{�{�нννнݽ���������ݽннннннн���
���"�/�;�H�L�H�>�;�/�"����������������������������������������������������������������������� ��������������FFFFF$F1F9F=FIF=F1F$FFFFFFFF���������������� ��������������������������� �����������������򹪹��������Ϲ׹ܹ�����������ܹչù����Ϲʹù����������ùƹϹعܹܹܹӹϹϹϹϿѿǿĿ��������Ŀѿݿ������������ݿѿG�C�E�D�G�M�T�m�y���������������y�m�`�G��������������������������������������f�M�1�(�#�,�(�*�3�@�M�Z�f�{���x�{�x�s�f�����������������ûм���������ܻû��������������	��������	�������������	��"�'�6�0�.�,�"��	�����N�B�5�.�)�$�&�)�5�B�[�p�t�w�t�r�g�[�R�N������%�(�4�A�J�M�P�M�A�=�4�(���²ª§¬²½¿������������¿²²²²²²�l�_�S�F�D�<�F�H�S�_�l�x�{�����������x�l�s�r�g�d�g�s�������������s�s�s�s�s�s�s�s����s�n�o�������������������������������������������Ŀѿݿ���ݿݿѿĿ��������'������'�2�4�@�M�T�O�M�I�C�@�4�'�'�Y�e�r�~�����ټ��������ּ���v�f�Yààßàìîùýùìàààààààààà����ĿĿ����ĿĽĿ��������"�%�&�����̹�߹�����������������������Ϲ̹ù��¹ùϹعڹչϹϹϹϹϹϹϹϹϹϺL�C�@�=�@�I�L�T�Y�e�r�~�������~�r�e�Y�L�������������Ⱥɺʺֺܺغֺɺ������������U�K�H�E�F�H�U�a�n�zÇÈÓ×ÉÇ�z�n�a�U�H�?�/����#�/�H�n�zÓÞØ�z�n�a�U�K�H�.�,�$�.�G�S�`�������������y�l�`�S�G�:�.E*EE%E*E7E=ECEPE\EfEaE\EXEPECE7E*E*E*E*EuEpEsEuE�E�E�E�E�E�E�E�E�E�EuEuEuEuEuEu��
�����'�0�4�@�@�E�@�4�'����� ? ` 9 N & B 0 I 8 � T : p t 5 ) K N P J 2 U A ` f : r / u [ \ > | N M @ 7 = D h X h K 6 J [ @  1 F B e Q  ; S ? ; C n 0 E p i ? [ N 2 0 ^ f y O r /    t  �  �  �  2  O  �    �  J  C  �  �  N  �  �  =  *  �  �  )  "  �  �  �  �  N  5  �  �  b  �  �  �  e    �  #  j  �  o  t  y  �  D  �  �  
  �  _    �  w  �  �  �  �  F  K  �  V    )    y  l  6  �  �  Y  �  H    �  ٽt��ě���9X��`B������o�D���49X�D�����ͼ49X�e`B�]/�o�@���h�49X��t��P�`��󶼋C�����/��t���t��+�H�9�49X��1�t��o��P��j���8Q�0 Ž�\)���+�o�����0 Ž,1���H�9�,1�q���ixս,1�q����\)�aG��aG��e`B�T���y�#�ixս@�����y�#�aG�����T�����P�ixսm�h��O߽�7L��hs��񪽼j��xս�Q�)��B�vB��B,#B�?B'&�Bv�B��B=B��B��BFyB$7'B�B!S�B�B�B4�B&j�B+,B ��B��B!�B��Bk�B!�B4�oB�[B'�`B�BaBB��B0M<B�9Bp�B��B��B =�B�B!��B{B
�jB��B�BI�B�B  BkB
�aB�_B �B;�B${�B	��B�FBR�B%�B�AB"�B��B��B��B)t�B,�oB"Bv]B�:BB B!x�B ��BۄB
q�B?�B�BC�B�BOQB��B>�BE;B'aBA9BG�B4�BA~B��B?B$@B�B!~�B��B�?B?<B&N�B*��B!;�B��B@�B@�B��B!=�B4��B��B'K�B<QBCKB4�B0@#BҮB?�B�5B)A��B	�B!��BŃB
�BBGB�BG<B;�BB�BL�B;B�XB ��B�B$��B	�B6�B?�B%�\B3YB"L{B��B1�B��B)��B,�B�fB�B�B?uB"#�B!�B��B
��B��B:�BAdB?�@��A�	Ar�(Aa6�AX�A>AG�AF��A���A�f�A�&TAP{�B[A�zzAp�A���A>\�A��Ax��AN��A�Z�AUc�@@
'A�q�A<�;AL��A���@��_A��:Bm@�AV��A��AX>�A�VgA�+�A�܈A���A+�A�F�A�.�A�4qC��`AӟB�>���>U��A|5�Aj�]A���A?��@�4VAZ�?A[��A�vA7��A�A\@�@gA�5�A�'�Az%3@Ξ@��MÅ�A�ϓ?��>�yB?�6�@-;A�5kAĞ�A�C���C�>@ȍ@�cA�e�Atv�AaRAFTA~�AG
^AG�A�v�A׀)A�}gAOCB	8�A��Ap��A�|�A=6A�yGAv��AM
�A��AQ�]@G�fA�R�A>�AMoA�6�@t3sA�?B�@��AW*�A�|�AW	�A�t�A��]A�O�A�&DA,�ZA�C-A���A�wFC���Aӄ�B��>�R>M��A}�Ak�A怖A?6@�
@AZ��AZ�2A��A9sA��@�7A���A�}1Ay�^@�$A˪A��UA�~??*M>�?�o@/ AƁ?AĊ�A�C���C��@�I�   ,                  	      	         	   1      )         B   -            b            "                              /      	               
         
               !                              
   &         	   	            *      !   	   #                                       A      '         A   7            E            '                        +   !   !                                          %   /                        +         /      )                  !                                                   ;      !         ?   5            ?            %                           !                                             %                           %         %      )                              N��/N��O�VDN�#O�,PN0�ENl��N�njN�ɂO?#2Nl�N��P|�$N,POܭ�O:� N#;P�?hP-�mO1�N��6Ow<�P�0�N���N���NLO�a�On��N�˘N�aTO�0O,��N8N��O��SO�A�O��	O�N`/4N���N:SNE�NP=jN$wN�0O,��N�*XO���N�	�N6��O�^O)jfN�k�OR�~O)*�N�G�N�3�N�yN8'�O�i�O&N�$�O�EN�P
�N#��M� POlW�N�R�O	��O$ vOH��N�$�Nl		N�ۄ  
  �  i  �  �  �    �  Z  �  �  �    �  $  �  o  �  �  R    �  _  7  #        �  �  D  X  S  h  p  -  �  f    i  �  �  Z  s  U  �  �  �  �      `  R  �  �  �  �  [  $  �  �  �  �  s  W  �  `  	  �  �  @  F  
M  @  ��o%   �ě��o��o�D�����
���
��o�ě����
�ě��D���ě��T����`B��`B�#�
�T���D���D���D����j�e`B�e`B�ě���C����㼓t����
���
��9X��1��1��/��9X�����ͼ��ͼ�����/�o�o�t��+�+�+�\)�0 Žt����P�`�0 Ž�w��w��w�,1�,1�0 Ž49X�49X�<j�P�`�@��D���H�9�L�ͽ]/�]/�q�����㽟�w��{��1'����������������������������������������47?BEO[hlrzzth[PB=548BCOS[`hkhee[PONB:886<IUbowsqidUKI<86646��������������������" �ABKN[^^][WNB69AAAAAA����

����������
#0DJHA<7/
����"#/;<C</#"""""""""""

����� 
mz����������zlfm�������������������������		���������� �����������������������������0In{�����{UI<���z�����������������zz��������������������[[dhtx�����th[VU[[[[.038<H]hicXTJ<</%'+.nz���������������gdn&)5;BCKNONLB:52-)'&&����������������������������������������29BN[t����|ywrg[N=42DIUbkwzz�~qbUQIGGCAD����������������������������������������")06BCOQSYOB6)*67COQSSIB6*'�������������������������&).5CLSgebZNB2(##')&��������������������moz����������zmcabem���������������z}����������������������������������������}��������w}}}}}}}}}}������������������������������������������������������������������������������\agnz�������zsngaZY\��������������������otv�������������yrno����������������������������������������!)/6BO[htusnh[6	�����
#'/04,#
����T[fgtu�������tga[TTT��������������������������

�������#$02<>C@<80-%# ��
" 
���������������������������������������������������)AO[ht�th[B)���������������||����t{}������������{vttt������������������������������������x�����������������~x$)6BKBB6)'$$$$$$$$$$MOX[hijh[OMMMMMMMMMM����������������������������������������358BNXYSPNONDB543223rt|������������ytqpr����������������
#$(#"
'/<@<<<82/)%(+''''''�������������軪�����������ûлܻ������ܻлû������/�-�'�/�<�H�P�U�_�]�U�U�H�<�/�/�/�/�/�/�����y�o�m�j�j�y�������ĿʿͿʿĿ��������.�"�"� �"�%�"��"�*�.�;�E�G�I�G�D�;�.�.�����������ʼּ������������ʼ�������������������������������������������������������������������s�q�n�s���������������������ùõìëèéìù����������������ùùùù���
���#�)�6�B�O�[�_�[�Q�B�7�0�)���H�H�F�H�I�U�W�^�X�U�H�H�H�H�H�H�H�H�H�H�˾׾۾���׾׾ʾ��������������˾˾˾�ơƔƓƝƹ���������$�0�>�<�<�2������ơ���	���	����"�#�"�������������y�h�Z�Y�`�m�y���������Ŀ˿ƿſ�����¦¢¦²¿��������������������¿²¦�M�L�A�M�Z�f�n�f�Z�Z�M�M�M�M�M�M�M�M�M�M�������w�[�K�6�5�F�g�������������������׿Ŀ����u�j�k����������ʿֿ��������ľ������������������ʾӾ׾޾���׾ʾ����;�:�/�)�)�/�:�;�H�T�T�X�\�X�T�H�;�;�;�;��׾ʾ��������ʾ׾����	����	����㺨�������p�e�������ֻ�-�I�P�F�!���ɺ����s�r�n�s�~���������������������������s�s�4�+�(�4�@�A�A�M�Z�\�f�h�n�f�a�Z�M�A�4�4�����������������ľ����������������������5�)����������������5�B�O�W�W�N�B�5�!����!�-�:�F�S�_�l�u�{�x�d�S�F�:�-�!�5�,�+�(�'�(�5�:�A�N�Z�b�Z�N�L�C�A�7�5�5ǈǄ�{�o�b�V�R�V�b�f�o�{ǈǔǖǘǔǉǈǈ�S�P�S�V�S�Q�R�S�_�l�x�������y�r�q�l�_�S����׾˾ʾľǾʾ׾���	�����	����ŠşŠũŭŵŹ��������ŹűŭŠŠŠŠŠŠ�������׾Ӿ׾�����	�����	�	�������������������*�6�C�M�O�C�6�*���������������*�6�C�P�U�X�R�O�M�C�#������������(�5�A�H�O�U�W�N�A�5�(��{�z�n�j�b�U�b�n�p�{ŇŔŜŝŘŘŔŇ�{�{�нννнݽ���������ݽннннннн���
���"�/�;�H�L�H�>�;�/�"����������������������������������������������������������������������� ��������������FFFFF$F1F9F=FIF=F1F$FFFFFFFF������������������������������������������ �����������������򹪹��������Ϲ׹ܹ�����������ܹչù����Ϲʹù����������ùƹϹعܹܹܹӹϹϹϹϿѿǿĿ��������Ŀѿݿ������������ݿѿ`�Y�T�S�T�T�`�m�y����������y�m�`�`�`�`��������������������������������������f�M�1�(�#�,�(�*�3�@�M�Z�f�{���x�{�x�s�f�лƻû������������ûƻлܻ����ܻۻо�������������	��������	�������������	��"�'�6�0�.�,�"��	�����N�B�5�.�)�$�&�)�5�B�[�p�t�w�t�r�g�[�R�N������%�(�4�A�J�M�P�M�A�=�4�(���²¬¨®²¿����������¿²²²²²²²²�l�h�_�S�I�F�A�F�N�S�_�l�t�x���������x�l�s�r�g�d�g�s�������������s�s�s�s�s�s�s�s�����s�o�o�������������������������������������������Ŀѿݿ���ݿݿѿĿ��������'������'�2�4�@�M�T�O�M�I�C�@�4�'�'�����������������������㼽������ààßàìîùýùìàààààààààà����ĿĿ����ĿĽĿ��������"�%�&�����̹�߹�����������������������Ϲ̹ù��¹ùϹعڹչϹϹϹϹϹϹϹϹϹϺL�C�@�=�@�I�L�T�Y�e�r�~�������~�r�e�Y�L�������������Ⱥɺʺֺܺغֺɺ������������U�K�H�E�F�H�U�a�n�zÇÈÓ×ÉÇ�z�n�a�U�/�,�#��#�%�/�<�@�H�U�]�n�z�n�a�U�H�<�/�.�,�$�.�G�S�`�������������y�l�`�S�G�:�.E*E E&E*E7E@ECEPE\E\E_E\EWEPECE7E*E*E*E*EuEpEsEuE�E�E�E�E�E�E�E�E�E�EuEuEuEuEuEu��
�����'�/�4�?�?�4�'������� 7 ` 6 N & B * U 8 � T : q t . ) K N N G 0 R @ ` f + p , u [ \ C | N 6 @ $ = D h X h K 6 J [ @   F B 8 H  ; S C ; C j 0 E U i ? [ N 2 0 ^ K y K r )��     �  :  �    O  �  �  �  7  C  �    N    �  =    n  �  �  �  �  �  �  _    �  �  �  b  �  �  �  @    S  #  j  �  o  t  y  >  D  �  �  
  �  _    �    �  �  �  �    K  '  V        y  l  6  �  �  Y  s  H  �  �  �  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  ?<  	�  	�  
  
  
  
  	�  	�  	�  	i  	1  �  �  l    �  C  �  c  �  �  �  �  �  �  �  �  �  t  J    �  �  M  
  �  �  E     �  V  `  f  i  i  a  Q  >  (    �  �  �  �  Q    �  �  "   �  �  �  �  �  �  �  �  |  u  l  d  \  K  6     
   �   �   �   �  �  �  �  �  �  �  �  �  �  �  }  S  %  �  �  �  a  .  �  �  �  �  �    ^  =    �  �  �    Z  6      
    �  �  �              �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  r  k  f  a  a  e  j  d  [  R  Z  Y  X  V  T  R  M  G  A  :  0  $      �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  Z  6    .     �  �    _  �   �  �  �  �  �  �  �  �  w  h  Y  J  :  )      �  �  �  �  �  �  �  y  s  o  i  ]  R  @  -      �  �  �  �  �  �  n  Y  �           �  �  �  �  m  G     
  �  �  X  �  X  r  6  �  �  �  �  �  �  �  x  p  i  a  X  O  F  =  4  ,  #      �    "  #      �  �  �  y  E    �  f    �  D  �  �    �  �  �  �  k  S  <  "    �  �  �  ]  ,  �  �  �  4  �  �  o  m  j  h  e  a  X  O  E  <  1  %        �  �  �  �  �  o  �  �  �  h  1  �  �  H  �  {    �  p  #  �  �  @  �  N  �  �  �  �  �  �  �  �  f  I  (  �  �  �  :  �  �  ;  �   �  2  H  Q  R  P  K  F  <  /       �  �    F  	  �  H  �                      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  k  a  e  T  #  �  �  2  �  �  1  I  ]  [  G  )    "  :  2    �  �  j    �  �    �  k  7  +        �  �  �  �  �  �  �  �  ~  m  f  b  _  [  X  #        �  �  �  �  �  �  �  �  �  �  �  �  �  ~  n  _  �  �  �  �  �  �                �  �  �  �  L  �  x  
      �  �  �  �  �  �  �  �  f  =    �  �  $  �  �  i  �      
  �  �  �  �  �  �  �  �  u  4  �  �  G  �  �  &  �  �  �  �  �  �     �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  W  ]  l  p  `  M  4    �  �  �  b  .  �  n    �  Q  D  3  $    
  �  �  �  �  �  �  k  R  8    �  �  y  �  O  T  T  W  V  R  J  >  -    �  �  �  �  l  <    �  {       S  A  /      �  �  �  �  �  �  �  �  �  
     5  J  _  t  h  W  E  ,    �  �  �  �  V  )  �  �  �  p  H  4  &  >  Y  /  C  S  _  n  o  j  _  O  8    �  �  �  �  g  0  �  �  m  -           �  �  �  x  K      �  �  �  �  �  |  ]  �  C  �  �  �  �  �  �  �  _  /  �  �  z  3  �  a  �  Y  �  /  f  \  R  I  ?  6  -  #        �  �  �  �  �  �  �  �  �          �  �  �  �  �  �  �  �  n  W  @  &    �  �  �  i  ^  R  G  9  *      �  �  �  �  �  �  �  t  P  �  }    �  �  �  �  �  �  �  �  �  �  �  x  g  U  D  4  %       �  �  �  �                     !      �  �  �  �  a  Z  K  ;  *      �  �  �  �  �  �    e  K  9  .  ,  V  �  V  _  i  l  o  q  s  m  d  R  :    �  �  �  M  �  �  M   �  U  G  9  ,        �  �  �  �  �  �  �  �  �  �  �  �  y  �  �  r  a  M  7       �  �    D    �  �  �  �  T    �  �  �  �  �  �  |  g  M  1    �  �  �  �  j  I  6  '  ;  O  �  �  �  �  �  {  g  P  7    �  �  �  {  M    �  5  �   �  �    4  f  �  �  �  �  �  �  �  �  �  X    �  _  �    Y      �  �  �  �  �  �  �  �  x  q  j  b  [  T  M  F  >  7      �  �  �  �  V  !    �  �  �  �  �  �  �  y  [  G  U  �  �  �  �  �     F  Z  ]  M  /    �  �  �  @  �  �  �  8  ,  8  A  I  O  O  E  1    �  �  �  �  H  �  �  J  �  a  a  �  �  �  �  �  �  �  �  �  �  |  e  I  &  �  �  �  f    z  �  �  q  [  H  7  &      �  �  �  e  ;    �  �  V    �  �  �  �  o  R  4    �  �  �  �  �  Y  ,  �  �  q  1  �  o  �  �  �  �  �  �  �  �  j  J    �  �     �  -  �  u  J    O  Y  Z  T  G  6  %      �  �  �  �  �  e  B  $    �  �  $    	  �  �  �  �  �  �  �  }  p  d  W  J  =  0  "      �  �  �  �  �  �  �  �  �  o  J  %  �  �  �  ?  �  �  !   �  �  �  x  i  X  C  +    �  �  �  �  X  $  �  �  X  	  �  k  �  �  �  �  �  t  e  Z  Q  D  3      �  �  �  �  �  �  �  �  �  �  �  y  M    �  �  r  8    �  �  �  L  �  b  �   �  s  }  �  �  �  �  x  j  [  L  =  /      �  �  �  �  �  �  W  F  '  �  �  �  �  �  �  �  �  q  ]  K  =  '  �  �  q  f  �  �  �  �  �  �  �  �  �  �  �  �  �  v  b  M  7  2  G  ]  `  `  `  `  _  Z  I  7    �  �  �  �  p  M  )    �  �  �  	  �  �  �  �  �    _  =    �  �  �  �  p  K    �  �  :  �  �  �  �  �  �  �  �  x  a  H  0    �  �  �  �  �  ]    �  �  �  �  �  �  �  T  "  �  �  }  X  1    �  �  �  u  R  E  h  �  �  �    ;  >  .  	  �  �  K  �  �  Y    �  F  �  F  4  $      �  �  �  �  �  �  �  l  Q  7    �  �  s  <  

  
6  
K  
?  
-  
  	�  	�  	�  	T  	  �  �  3  �  �  &  R  t  �  @    �  �  �  �  p  L  C  >  (     �  �  �  S  !  �  �  \  
�    
�  
�  
�  
c  
*  	�  	�  	6  �  S  �  ,  [  �  �  �  �  