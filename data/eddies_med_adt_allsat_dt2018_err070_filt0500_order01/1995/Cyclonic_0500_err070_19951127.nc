CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�bM��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �C�   max       =+      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?O\(�   max       @F�\(�     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @v33334     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P`           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @���          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �,1   max       <���      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��E   max       B4�%      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��L   max       B4�N      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�\r   max       C���      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�ƻ   max       C���      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          _      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          K      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          K      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P���      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�t�j~��   max       ?��1���.      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �C�   max       =+      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?s33333   max       @F�\(�     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @v|(�\     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P`           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @���          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A~   max         A~      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?x*�0��   max       ?��1���.     �  V�         E   =                     &      	               
      "   6   1               ^            $   
   )                     '   .         4   ,      2                  ,      6                        !N&�N�>�P�bPY��N��~O��vNM�Np �N�*�N;3�O���N�(N`�P_O#/�N5ANO��N4��N�tnPoÉP��3PoK�P��NF��N�C�M���P���O��kO�&N��O��BNhC�O��aO"*�O�JO��YO��`O��N��P���O��{O��tO�P$��Ot�sOZ��Oҍ]OK�N6IRO��ROH�O�6O�޸O'�P;�+N�%N�#N_(�Oy��O{�O�zO)��N�0n=+<�/<�C�<e`B;�`B;ě�;�o;o;o:�o��o���
���
���
�ě��ě���`B�t��e`B�u�u���㼛�㼛�㼼j��j���ͼ��ͼ��ͼ�/��`B��`B��`B�o�o�t��t��t���P���,1�8Q�@��@��@��D���H�9�P�`�T���T���Y��Y��ixսixս�C���C����P���w���w���署-����C��������������������������������������������������������������������%/:$�����!#/6<=B<://.# !!!!T\\[[an{������naUOSTbhotx����tkhbbbbbbbb�������������������

�������������������������������#<HRVPHB</#V[ghng_[YUVVVVVVVVVV�����������������������
"CVbh{uh\C*���������������������������������������������)5;?:3)�����:BOU[^[SOMB?::::::::��������������������;HTamz��������aWH??;���#<n������yT<0���wx}�������������nfw>B[t����������t[H@=>������������������������������������������������������������n������������{qpvlfn+6;>BOdpjba^[O6)����

��������)/>BNB5)����������������������������������������������������������`aegmz������~zma`__`������ ')"���������������� ����������������������������������� �����������������������������#0Rej{�������{fI<##hkt������������t]]bh')BNSZffgng[N5.)&).'/<HMQVVUSZaaUK<8/'&/JUan���������znaHCFJ{�����������������~{^hnz���������{znja[^ACO[htx�����thbEAA����������������������������������������GO[htz}}tlh][YZWRIEG##+/AHTU[ahjaZUH</*#�������		
�������)4BNZNGB2)���S[]gt����������td[TSEKT]gt����������gNBE��������������������x����������uxxxxxxxx���������������������������������8<IMUWYYXVPI<5124788#%./<AB@<4/-#"ktx�����������toihkCHMUaalkbaUKHH>>CCCCE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�¦¯²¶µ²®¦�/�#�����#�<�U�a�n�yÅÆÄ�~�n�a�H�/�������s�Z�A�5��3�A�b�s�����������������<�9�1�<�D�H�U�^�a�n�n�n�m�a�U�H�<�<�<�<�������s�Z�B�I�Z�s�����������������������H�D�<�5�<�D�H�U�_�^�U�M�H�H�H�H�H�H�H�H��������������	����������������������EEEEE*E1E7ECELEKECE7E6E*EEEEEE�l�b�f�`�l�w�y��}�y�l�l�l�l�l�l�l�l�l�l�6�)�"�!�"�$�)�6�O�Z�h�p�t�q�u�t�[�O�B�6�f�a�e�f�s�{�����s�f�f�f�f�f�f�f�f�f�f�H�B�<�8�<�H�K�U�`�[�Z�U�H�H�H�H�H�H�H�H�T�M�.������־����.�;�D�G�E�H�L�]�TàÓÒÐÎÑÏÓàäìòù��������ùìà�����������ʾվϾʾ����������������������s�Z�V�W�\�Y�U�Z�g�s�������������������s�a�X�\�a�j�n�u�zÁ�z�q�n�a�a�a�a�a�a�a�a�H�G�B�C�E�H�M�U�Z�a�d�n�n�n�m�g�W�U�M�H��������������������/�;�K�L�S�H�#��������������^�H�4�1�A�Z��������������������<������������������
�#�I�b�~ŌŇ�{�b�<���۾Ǿ��������׾����.�3�1�2�.����𾘾��������������������������������������Y�T�M�J�B�C�M�U�Y�^�f�k�p�r�r�r�g�f�Y�Y�Y�W�Y�_�f�p�r�s�r�r�g�f�Y�Y�Y�Y�Y�Y�Y�Y��������'�@�e�����ܺӺ������e�L�@�'��������������x�t�v�����������ûƻλɻ��������������������������ü˼ּּ�ּʼ������m�j�l�m�q�v�y���������������������z�y�m����Ʊƫ��������������$�-�5�2�*������ĿĿ����Ŀѿڿݿ�ݿ׿ѿĿĿĿĿĿĿĿĻx�t�a�V�M�F�D�F�_�l�x�����������������x������߿���������(�-�4�+�(��������������ɼ�����!�.�.�#��	��㼽��ŹŴŠŔŊ�{�y�}ŎŠŭŹ��������������Ź���Q�F�@�I�S�_�x���������ûλȻû����������ܻлû����ƻлܼ��'�,�0�.�'����������y�s�t�y�������������������������������������S�A�Q�����Y�i�u�c�'�����л�ŭŠőńŃōŠŹ�������������������ŭ�������������*�3�C�H�I�N�C�6�*�����E�E�E�E�FFF$F-F1F=FIFKFJF=F5F1F*FFE��T�>�3�-�/�<�H�a�m�������������������m�T�Ŀ������������Ŀѿݿ�����������ݿѿľ����������������ʾ׾��������׾����Ϲù������������Ϲܹ����� ����ܹ�������������������$�*�.�0��������������������������������������������������������������������#�/�:�/�"���������������������������������ĿοѿҿпοɿĿ��ɺú������������ɺ�����������íÝÙØàïîùý��������������������í�������������	�	�������	���
�����p�l�n¿�������
���*�$�
���
��������
���#�*�.�#������ìåàÝ×àìùý��üùìììììììì�����!�.�3�:�F�>�:�.�-�!�������l�b�S�I�S�a�y���������ɽĽ����������y�l�нǽȽнڽݽ������%�&�������ݽ�DbD`DbDoDpD|D�D�D�D�D�D�D�D�D�D�D�D{DoDb������ĿľĹĸĿ����������������������E�E�E�E|E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� f L  H 7 D R F f _ , 8 P H I 7 M M f 8 g I F H D � P C G � L 3 H B q 7 p 2 C e 1 7 i ( " / > p e ~ 0 B ? d [ D R Z ] R O I 2    i  �  f  �  �  �  {  �  �  ~  V  #  �  �  y  P  �  j  :    �  �  �  a    O  �  *  E  k  *  v  s  ]  S  ,  �  �  .  �  1  C  �  �  �  �  �  r  y  �  �  �  �  �  �  �  �  �  @  f  H  }  �<���<T���T���@����
�D��%   �D���e`B�ě��'o�e`B�t���j�#�
��h������D����C���7L�8Q켬1�0 ż��ͽ�F�49X�0 Ž+��%��P��O߽8Q�q����o�Y��ixս�w���-��9X��%�y�#�ȴ9��^5�u�ȴ9�m�h�u���w����������������ٽ��T��Q콩�罾vɽƧ��/��/�,1B�9BvB>�B�xBRBABD�B��B)/B+1�B!0B�B!�B0�B��B4�%Bq�B��BG B ^�B%�VB]UB;aB!9�B �mB#%SB*B��B$�B�%B�	B�CB!�A��EB-��B{�B��B��B*�lB(�2B
�HB�B�BF�B�GB;�B�"BͦB:B��B+RB#)CB#�B	��B	��B´B
�BB7�Bs�B&��BIB
o�B�&B8B;nB>B�cB?�B@Bw�B��B=�B+g2B"�B�_B!>�B/ƪB��B4�NB��BFJB8vA��-B&��B��BBiB!A�B ��B#;B?�B~=B$9;Bz�BƳB�B �}A��LB.>�B��B��B�B*��B)@�B
��B�B�]B�FB:�B`BE|B�DBA5BHB;�B#>�B0�B	�,B
C�B�EB
�B.=B��B&��B�B
@�B�.C�1�A��A��DA��XA�+4A�-�A���A�Y"C��`A��A��YABͤA��A[O4A�<AOfA�`YA�AAŠA��MA���A��AY@�AIz@�z@�+?�	�@��@��Am�BC#Aze�@�1�A�3�A�%A���@��@��Ane�@�6�A��qA��C���A��FA{9�AP�$>�\rB�A�A���Av\�@>�AΜNAY�A�f�A�(A̝3A)�A�.A/f|C���A���C�/C�8�A�!A�*TA��mA�~�A���A��
A��C���A��A��ABF-AĀ�A[ A�4AN	A�v�AƂbA��A�/�A�lA�w�AY�AJ�&@�	@�G?�k'@��t@���An�QB�mAz��@��A�XuA)�A�u�@�j�@��vAnU.@��OA���A�xnC���A���A{)4AN�W>�ƻB�A��A�y�AuK@4d�A΅�AY�A���A�yqÀ�A +A�A1q�C��iA�4 C��         F   >                     &      
                      "   6   1               _            $   
   *                     (   /         4   -      2      	            -      6                        !         #   3      #                        -         #         3   C   3   )            ;            '            '               K   %         )         %         %      %         1                                          #                                 #         )   C   1               7                        %         !      K                     %         !      %         #                        N&�N�>�O�*�O�`UN��qO��vNM�Np �N�*�N;3�O\5�N�(N`�O8UN�@N5ANO��NT�No\�P)A�P��3PB�dOȑNF��N�_�M���Pl��OK-�Nզ�N��O�&�NhC�N�H}O"*�O��]O�c�O��`O���N��P���O[��O���N�cOĴ�O?�OZ��Oҍ]OK�N6IRO�UKOH�O�6O�jkOΟO�ņN:�N�#N_(�OL�O[�O�zO)��N�0n     ]  	  �  �  6  "  �  �  �  B    n  x  �  �  W  n  (  4  \  �  �  |  �  $  �  %  �  	  w  �  Z  �  �  �  �  �  �  �  �  �  )  �      X  7  T    �  �  x  �  �  �  R  �  �  _  �  �  �=+<�/��o�e`B;��
;ě�;�o;o;o:�o�T�����
���
��t��t��ě���`B�#�
��t���9X�u���ͼ�j����ě���j��P��`B��/��/����`B�8Q�o�+��P�t���P��P���q���<j�D���y�#�]/�D���H�9�P�`�T���Y��Y��Y��}�u���
��\)���P���w���
��1��-����C�����������������������������������������������������������������

�������##//:<><7/#"######T\\[[an{������naUOSTbhotx����tkhbbbbbbbb�������������������

�������������������������������#<CHMMJH<</#
V[ghng_[YUVVVVVVVVVV��������������������$*6CLOOJCC76*��������������������������������������������)5;?:3)�����;BOT[][OOOB@;;;;;;;;��������������������Tmz�����������maQIJT���#<n������yT<0���m}��}������������wmmBHYht���������t[OFBB������������������������������������������������������������t��������������~w|ut'/69=BFO[^ih`_\OB4*'��

	���������)/>BNB5)����������������������������������������������������������`aegmz������~zma`__`����� &( ����������������������������������������������������������������������������������#0Rej{�������{fI<##ott������������}spno')1.BNQXddfkg[NB5.*''/<<HLPUUUUPH<9/-(''PVanz���������naUNLP��������������������^hnz���������{znja[^ACO[htx�����thbEAA����������������������������������������GO[hty|}|tkh[XXSOJFG##+/AHTU[ahjaZUH</*#�������		
�������)-4B@75/)����U[agt��������tjg[WUU^bgt����������t[SSX^��������������������x����������uxxxxxxxx��������������������������������9<>ILUVYYWUOI<623589#%./<AB@<4/-#"ktx�����������toihkCHMUaalkbaUKHH>>CCCCE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�¦¯²¶µ²®¦�H�<�/�$���%�/�<�H�U�b�n�x�y�w�n�a�U�H�s�g�W�L�G�J�N�X�g�s�������������������s�<�<�;�<�H�M�U�Y�a�k�f�a�U�H�<�<�<�<�<�<�������s�Z�B�I�Z�s�����������������������H�D�<�5�<�D�H�U�_�^�U�M�H�H�H�H�H�H�H�H��������������	����������������������EEEEE*E1E7ECELEKECE7E6E*EEEEEE�l�b�f�`�l�w�y��}�y�l�l�l�l�l�l�l�l�l�l�6�-�)�%�'�)�.�6�B�O�Q�h�i�m�g�h�[�O�B�6�f�a�e�f�s�{�����s�f�f�f�f�f�f�f�f�f�f�H�B�<�8�<�H�K�U�`�[�Z�U�H�H�H�H�H�H�H�H������������	��"�"�.�/�3�.�"��	��àÕÔÕÞàìîù����þùìàààààà�����������ʾվϾʾ����������������������s�Z�V�W�\�Y�U�Z�g�s�������������������s�a�Y�]�a�l�n�s�z�{�z�p�n�a�a�a�a�a�a�a�a�H�F�H�H�Q�U�^�a�c�j�c�b�a�U�H�H�H�H�H�H���������������	��/�;�D�H�H�K�H�/��	�����������^�H�4�1�A�Z��������������������U�<�#����������������#�I�b�{ňń�{�o�U�����پϾ;׾����	��)�-�,�,�!�� �������������������������������������������Y�U�M�L�C�D�M�Y�f�j�o�r�f�f�Y�Y�Y�Y�Y�Y�Y�W�Y�_�f�p�r�s�r�r�g�f�Y�Y�Y�Y�Y�Y�Y�Y���������'�3�~�����ɺϺɺ��~�L�@�'������������x�x�{�������������û̻ʻû��������������������¼ɼʼѼʼ��������������m�j�l�m�q�v�y���������������������z�y�m����������������$�*�3�3�0�(��������ѿĿĿ����Ŀѿڿݿ�ݿ׿ѿĿĿĿĿĿĿĿĻx�s�l�c�_�_�_�b�l�x�����������������x�x������߿���������(�-�4�+�(������������ʼּ�����!�-�,�"����Լ���ŹŭŠŔŇ�z�~ŏŠŭŹ����������������Ź���Q�F�@�I�S�_�x���������ûλȻû����������ܻлû����ûɻм��'�+�/�.�'����������y�s�t�y�������������������������������������S�A�Q�����Y�i�u�c�'�����л�ŭšŠŔŔŜŠŭŹ������������������Źŭ��������������*�6�C�F�H�M�I�B�6�*�FF E�E�E�E�FFFF$F%F1F8F4F1F)F$FFF�T�H�<�6�4�6�;�H�T�a�m�z�����������z�m�T�Ŀ������������Ŀѿݿ�������ݿԿѿľ����������������ʾ׾��������׾����Ϲù������������Ϲܹ����� ����ܹ�������������������$�*�.�0����������������������������������������������������������������������� �������������������������������������ĿοѿҿпοɿĿ��ɺú������������ɺ�����������ñáÝÝàìù������������������������ñ����������������	��������	����¿�{�v�z²¿����������������¿��������
���#�'�#��
����������������ìåàÝ×àìùý��üùìììììììì�����!�.�3�:�F�>�:�.�-�!�������y�l�`�T�`�e�l�y���������Ľ������������y�ݽѽн˽н۽ݽ������$�%��������DbD`DbDoDpD|D�D�D�D�D�D�D�D�D�D�D�D{DoDb������ĿľĹĸĿ����������������������E�E�E�E|E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� f L  + 2 D R F f _ " 8 P / , 7 M S l / g J / H F � Q .  � 6 3 8 B l 5 p 5 C e  0 , .  / > p e q 0 B / H W 6 R Z X N O I 2    i  �  \  q  �  �  {  �  �  ~  �  #  �  x    P  �  W  �  �  �  d  �  a  �  O  *  �  �  k  �  v    ]      �  �  .  �  �    
  �  �  �  �  r  y  �  �  �    A  Z  [  �  �  �  M  H  }  �  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~  A~     �  �  �  �  �  �  �  �  �  �  �  q  _  @    �  �  �  {  ]  V  M  B  5  %    �  �  �  �  u  L  !  �  �  �  w  2  �  #  z  �  �  	  	  	  	  	  �  �  �  ^     �  �  4  H  +  T  @  �  �  9  v  �  �  �  �  �  �  �  l  '  �  �  �  S  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  h  o  k  `  @    �  6  5  ,       �  �  �  �  y  f  Z  I  >  5  &     �  �  B  "                �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  y  s  n  i  c  \  T  M  B  4  '    	  �  �  �  �  �  �  �  �  t  X  ;    �  �  �  �  �  ^     �  �  U  	  �  �  �  �  ^  5    �  �  �  �  ]  4  
  �  �  x  ;  �  �  �    #  7  @  ?  4  "    �  �  u  &  �  �  5  �  b  �  =          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  j  f  ^  W  P  I  A  8  ,      �  �  �  �  t  J    �  �  �  �    &  5  G  a  x  s  l  `  K  1    �  �  T  �    �  �  �  �  �  �  �  �  �  �  v  P  $  �  �  �  a  $  �  �  �  �  �  �  �  �  �  �  �  �  �    |  z  w  r  k  e  _  X  W  V  T  L  E  >  7  3  0  -  +    �  �  �  �  D  �  �    Z  e  n  k  g  `  X  O  F  <  1  '      %  G     �  �  i  �  �  �      '  $  
  �  �  �  I  �  �  W    �  �  h  0  �  �    ,  3  '    �  �  �  h  >    �  �  �  f  5  �  m  \  C    �  �  �  N    �  �  h    �  �  <  �  u    �   �  �  �  �  �  �  w  D  0    8  �  |  =  �  d  �  �  }    L  y  �  �  �  �  �  �  �  �  �  �  �  �  i  F  #  �  �  N   �  |  u  o  i  c  ]  W  Q  K  E  =  3  )            �   �   �  �  �  �  �  m  U  <  !    �  �  �  |  >  �  �  �  R    �  $            �  �  �  �  �  �  �  �  �  �  p  _  M  ;  �  �  �  �  �  �  p  \  L  /  �  �  2  �  6  s  �  �  �  �    
    %      �  �  �  �  �  �  �  �  j  B    �  �    k  f  �  �  �  �  �  �  e  G  &    �  �  x  2  �  e  �   �  	        �  �  �  �  �  �  �  d  H  0      �  �  �  �  �  v  v  j  T  <     �  �  �  �  B  �  �  N  �  `  �  L  �  �  �  �  �  �  �  �  q  Z  A  '    �  �  �  �  �  k  V  @  �  �    -  9  D  N  Z  Z  N  5    �  t  $  �  i    =  n  �  �  �  �  �  v  c  J  2    �  �  �  �  �  ]  )  �  �  L  �  �  �  �  �  �  �  �  �  b  1  �  �  e  3  �  �    `   Y  �  �  �  �  �  �  ~  [  -  �  �  d    �  �  d  3  �  �  N  �  �  �  �  �  �  �  m  M    �    �  �  �  b    �  8   �  ~    p  _  O  B  8  0  )      �  �  �  �  I    �  ^   ;  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  �  �  ~  O  *    �  �  �  u  K    �  �  �  E  �  �  3   L  �  �    \  �  �  �  �  �  �  q  D    �  b  �  {  �  �   �  �  �  �  �  �  �  �  �  �  �  v  U  '  �  �  �  �  �    
  �    #    
  �  �  �  �  �  �  o  M    �  �  �  A  �  u    F  t  �  �  �  �  �  �  m  <  �  �  _  �  �    w  �  }  �  �        �  �  �  �  s  G    �  P  �  `  �  7  .  z        �  �  �  �  �  k  J  2  #  +  4  ;  ?  E  S  �  �  X  I  G  8    %  "  #       �  �  �  �  q  N    �  �    7  '      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  T  T  S  J  <  +    �  �  �  �  b  9    �  �  �  V  %   �  �    �  �  �  �  �  �  �  �  k  ,  �  �  ?  �  �  e  l  J  �  �  v  R  &  �  �  �  t  C    �  �  N    �  j    �    �  �  y  Y  F  8  %    �  �  �  �  �  �  �  �  v     �  h  �  7  m  v  v  w  r  \  9    �  �  ]    �  �  D  }  �  o  �  o  �  �    l  T  8    �  �  �  F  
  �  �  Q    �  �  #  N    �  �  �  �  �  �  �  �  �  R  	  �  Y  �  q  �  j  �  �  �  �  �  �  �  �  �  g  I  *    �  �  �  q  I    �  R  C  0      �  �  �  �  Z  &  �  �  m  %  �  �  8  �  g  �  �  �  �  �  �  �  �  u  W  9    �  �  �  �  h  +  �  �  �  �  �  �  �  �  �  x  g  S  ;      �  �  �  �  �  �  t  ^  _  S  @  +    �  �  �  �  U    �  �  k  +  �  �  V   �  �  \  4  
  �  �  �  ]  /     �  �  ?  �  �  I  �  �  |  �  �  �  �  �  �  �  �  �  �  �  i  M  1    �  �  �  }  D  �  �  :  
�  
�  
�  
;  	�  	�  	N  �  �  4  �  U  �  -  {  �  �  �