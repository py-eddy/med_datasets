CDF       
      obs    A   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��l�C��       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��?   max       P�J�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��^5   max       =t�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?z�G�{   max       @F���
=q     
(   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vO��Q�     
(  *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @O�           �  5   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @�m            5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �J   max       <#�
       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��M   max       B4�}       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�_�   max       B4�       8�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C�=�       9�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >r��   max       C�U�       :�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          n       ;�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C       <�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =       =�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��)   max       P�t1       >�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�g�   max       ?�$�/�       ?�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��^5   max       <�h       @�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?}p��
>   max       @F���
=q     
(  A�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vO��Q�     
(  K�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @K@           �  V   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�,        max       @��            V�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D@   max         D@       W�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�	� �   max       ?�$�/�     P  X�            	      1            Z   H      m   8            B   V                     
      /         6            9   #   	            +      
      >   +      '                        )                  &   1   .   OJ;�N���N�N@�AM��?PD�^Ng=4N��hO��P�J�P��QN*�P���O�2N�hSN
��O8�4P
2HP���OD��O���PO&m3O�J�OUMO/�EN�a�P��OT�OM|APJ�pN��9OvY|N���P6�cPb�N�W�O�0O��N�AVO�gN0��N�yoO5�lP0�P$I!O�/�Oү.N�z�O2<�O�PF�OBz�N�g�Oe,�O��cN��qO1�$O%�9O0.NW,�O9'LO�i�O���OY��=t�<�1<���<�o<e`B<o<o;�`B%   %   %   �o��o��o��o�ě���`B�D����C���C���t����㼣�
��9X��9X�ě��ě���������������`B�������o�+�C��C��C��t������'49X�8Q�<j�<j�@��T���Y��]/�]/�e`B�e`B�u�}�}�}󶽋C���\)��t�����������
��^5#/1<BFE@<0/#
#*/8<:/###/0<CFIJQI<<0#�����������������������������������������)5NVU]cc]MB�����������������������LN[glt���}tpg[NMMILL���������������������#<U�������eI0����~���������}zst~��������������������{��������������|{)6AFU_ba[OB)NOR[hkttnh[POJENNNN����������������������������
�����������������������nbI<#�������
#<Ukwxn
/<HTXanonaUM<#

xz��������������~xtx��������������������ghnt�����������tifhg��������������������LO[hopmqroh`[OODCELLu��������������uwu|u��
#).-$#
	�����5N`kqn``[B5)��
#/;A><8/%#!
����������������)6B[t���ztiUOB92)#")��
#$.*#
������y{�������������}zzxy��������������������OUaz������naQHA9446Oz����������������|wz)67BHMB6*)).[qxytWOEHBA6!"6<?@BOZ\bnttn[B62326������������������������������������}{}�����������������������������������������GNY\gt���������tgZMG�����.0*�����������(?<6)
������������������������

����������������������������HN[gt{�����tog[PNGEH��������������������[gs{�����������{rne[RTZafmrz}zvmaVOLNR!)5;50)stw������������tqnpsmt������������~tifdmtt�����������vtikptt��������	���������+/4:<HLTRQQURHA<61,+T[\gt���������tg[ZST������������������������&)25/)����u�����������������yu��
(-('$
�����]anz����������tnaYY]����������������������������������������������
�����������������������ؼ��������������������������¼ʼʼǼļ����<�2�6�<�H�U�X�X�U�H�<�<�<�<�<�<�<�<�<�<����������������������������������������������ҿݿ����(�A�g�s����s�N�5�(��(��(�/�4�A�M�P�N�M�A�4�(�(�(�(�(�(�(�(���������������������¿ĿȿͿɿĿ��������M�A�F�Q�b�s���������������������f�Z�M���s�S�B�<�*�L�g������������
������˿��m�[�_�q���������ѿݿ�����'��ݿ�����¦ª®®¦�����������!�F�������̻û��_�F�-��!���"�-�:�S�l�x�����������y�l�_�S�:�!�Y�Y�Y�Y�`�e�n�r�{�~�����~�|�r�e�Y�Y�Y�Y��߿ݿֿܿݿ޿����������������l�e�_�[�P�L�W�Z�d�l�x���������������x�lā��x�z�o�uĉĚĦĳĿ����������ĽĦčā�N�S�M�>������н������|�y�|���Ľ��N����#�%���!��)�,�,�5�B�X�I�B�6���#���
�����#�0�I�U�]�b�i�k�b�U�<�#���
����*�O�h�uƎƚƴƼƩƎ�u�\�*��
����������������
����#�.�5�/�#��
�m�`�X�T�E�C�G�T�`�m�y�|�������������u�m�ù������ùϹܹ�������������ܹϹù��	���������	��"�'�/�6�;�=�G�;�/�"��	ìèãààÜàìù��������������ùóìì�/�"�����"�/�;�H�T�a�l�q�v�u�p�e�H�/�U�R�H�A�<�@�H�U�a�e�n�w�z�{�z�o�n�a�X�UàÞ×ÐËÈÍÓàìöù����������ùìà���y�w�x�p�x���������"�*�*�	�������������
�����$�0�6�=�=�=�;�1�0�$�����ùìçíù������������&������������������������������������������������ѾM�I�Y�o�������׾���	����׾�������M�*��� �!�$�/�H�T�Z�a�t�~�����q�a�H�;�*ìèçììöù����������úùìììììì��z����������˾���	���	���ʾ�������_�T�G�;�5�.�;�G�T�`�m�y�����������y�m�_������������������������������������������ػջԻܻ�����#�@�L�@�4�2�"������Y�O�M�Y�`�f�j�r�r�r�g�f�Y�Y�Y�Y�Y�Y�Y�Y�$� ��������$�)�-�.�$�$�$�$�$�$�$������������ �	�������	����r�T�E�=�9�9�@�M�r������ļʼԼ��������r� �&�"�*�S�y���ĽͽսٽĽ��������`�;�.� ���߽�߽�����4�8�8�0�$������н��Y�L�H�H�J�Q�a�e�r�~�����������������r�Y�����������������üʼּ�ݼ׼ּмʼɼ����*�$�#�*�2�6�B�C�O�[�^�^�\�X�T�O�N�C�6�*�V�N�I�@�F�I�U�V�b�o�{�ǈǈǈǅ�{�o�b�V��л��_�X�[�~�����л���@�Q�M�<�'���������������������	��"�,�/�2�5�/�"��	����	��������� �'��������ā�z�t�s�u�z�~āčĘĦĴĿ������ĿĳĚā¦¿���������������������¦�����#�#�/�:�<�?�H�R�K�H�G�<�/�#������������������������������������������D�D�D�D�D�D�D�D�D�D�D�EEE!EEEED�D���������������������������������������������������ùô÷ù��������������Ç�~�z�n�g�d�g�n�r�zÇÓØàãäàÔÓÇ�#� �
� �����#�<�I�P�R�U�_�e�Z�<�8�0�#�ݿĿ��¿ѿ����5�N�s�����|�s�^�A�(���ݹ��������������ùϹܹ�������޹Ϲù�  D ? - 0 D Y A O J 4 a C P L f P + G { 4 e . # 7 8 7 + # P 8 F d M R E n I ; " 4 8 = c % _ G 6 ] c 0 j = H < + ? ~ U K O " + y D    �  �    J    x  �  2  6  �  o  ]  '  �  �  V  �  q  �  M  Y    d    R  q    T  E  �  �  %  j  �  K  �  �  �    �  J  Y  �  �    �  k  �  9  �  ;  �  �  �  �      �  �  �  m  �  �  >  �<#�
<o<o;�`B<t��'�;��
;o�ě���E���hs�o��l��u�u�o��j���w���`�t����8Q���49X�@��+�\)��t��D���D����49X�m�h�49X��9X��7L�,1��o�ixսL�ͽ��T�<j�P�`�ixս�"ѽ�E���%�� Ž�%��O߽�t���{������hs��{���`���㽙�����
��vɽ�1��G��J�%���B�]BQB%�^B �qB�+B(�BqYB�B!��B&��B+b�B��B��B��Bm$B�B"R�BlB&B�BH[B �WB
B�*B��BS�B
�rB8�B�1B�PBP�B��B��B jB�ZBY�B��Bi�BsiB�DB4�}B�FB�B_B	�"BB	�BU�B"ϞB+��B	aCBX�B)�A��MBnyB
�cB
�pB
J�B��B�B	��B#B�gBB�lBF|B.�B?0B%��B �YBD�B=�BCiB	�B"@B&�}B+q�B�	B�AB�@B@\BJ"B"H�B�#B&��B�B �lB@VB�!B��B=zB8}B��B��B�	B�VB�B4�B �mBȩBŏBB:B�!B��B� B4�B@�BO�B@SB
mB?�B��B��B"��B,*UB�*B?�B*@�A�_�B?�B
~�B
@B
��B�XB�DB	��BDB��B5bB�KBA���A� �@��OAĂFA�U�A��aA9�KAv/�AEF-A�[�Aw��A�]�@�2#@�lc?�QA~th@�s?A��A+6`A�W�A��B��A���Ak�>�A�rA��A��AŶ"A�=�A�� B	��AЦA���ALx�A�l_A�^TAMصAkn�AKQ�@���@ۖ�B	z�AZ.@�WANdA0��@ AD@��B �,Bc�@�"A��A��5Aߣ�A��A¤@A�[5C�=�A���A��!A�_�A��A��>���A��A�wR@� IAĀ6A�}�A�}�A9h�Av)�AE hA��Aut0A��#@�6@��?�raA%�@�*A�y�A+�A׆�A�wEB N7A���Ai�>�?A��CA��A���A��A��A��wB	�YAτ�A���AJ�jA��Aͅ�AM Ak�AK)�@�t@��B	BmAZ�m@�pA��A1;?�\:@�r�B �B��@��WA���A���A��A���A��A��3C�U�A�e�A�b�A�m'A�dvA�e>r��            
      2            Z   H      n   9            C   V                     
      /         7            9   #   	            ,   	         ?   ,      (                         )                  &   1   /                     /            C   9      =   !            %   @         )                  #         /            1   '      )                     +   1   !   !            7            %                     #   -                                 =   5      1                  1         )                           +            1                              #   '   !               7            !                        '   O O�N���N�M��)M��?O�,�Ng=4N��!Oq�%P�t1Pm�N*�P?��O_lbN�hSN
��N��O~^P9i�N܎:O���POT1O5՜N�3FO/�EN_�OxomN�N��JP�rN��9Og�FN���P6�cOǾ�NC	O�>O4��N�AVOt#N0��N�yoN�#�O���Oߪ�O�/�O�QN�z�O2<�O�PF�O1ZTN�g�OW	�O�c�N��qN�I^O%�9O9NW,�O9'LO�H"O�jOY��  ^  �  ,  <  U  �  �  �  �    �  >  
+  �    �  �  =  O  M  �  X  �  }  �    �  o  �  �  c  j  �  �  j  q  �  _  c  �    �  �  P  �    �  \  �  �  �  e  �    �  B    B  �  �  �  
d  �  -  ^<�h<�1<���<e`B<e`B�u<o;��
�o�49X�t��o��㼣�
��o�ě��D���t��#�
��9X��t����㼬1��/���ͼě���`B�0 Žo�+��w���o���o�#�
�t��#�
��w�t��0 Ž��'@��ixս]/�<j�m�h�T���Y��]/�]/�ixսe`B�y�#����}󶽅���C���t���t������{���^5#$/3<=A?<8/##*/8<:/###/0<CFIJQI<<0#����������������������������������������#)5BNQSQNHB5)��������������������MNV[`gmtytge[SRNMMMM����������������������
#<n������{bH0��~�������������~w~����������������������������������������')6=BNSXYTOB4(  'NOR[hkttnh[POJENNNN������������������������������������������������������0U[dimlibUI<,#
�#/<FHNIH><7/# xz��������������~xtx��������������������gjhmst����������}tkg��������������������NO[hmnjnohhg[ROGFFNNu��������������uwu|u
#()#
)5BLNPNF=5)
	#/4;70/#�������� �����������)6[ht|~yyoh[O6/+'&&)��
#$.*#
������z|������������{zzyz��������������������OUaz������naQHA9446O������������������~�)26ABFB6.)$)6BO[ksuk[OIB6)#""$5:ABLOP[]hooi[OB@765��������������������~����������������}~����������������������������������������X[gkt��������tsg`[XX������%$!���������#)286)��������������������

������������������������������HN[gt{�����tog[PNGEH��������������������[gs{�����������{rne[ST\ampz}~~ztmaWTPMOS!)5;50)t{������������tropqteot��������������tgett�����������vtikptt��������

�������+/4:<HLTRQQURHA<61,+V[^gt��������tg][TVV������������������������&)25/)���������������������������
&*+'&#
������]anz����������tnaYY]�����������������������������������������������
�����������������������ؼ��������������������������¼ʼʼǼļ����<�:�9�<�H�U�U�V�U�H�<�<�<�<�<�<�<�<�<�<������������������������������������������������(�5�A�N�\�b�b�\�N�A�5�(��(��(�/�4�A�M�P�N�M�A�4�(�(�(�(�(�(�(�(���������������������ÿĿȿĿ������������L�J�M�U�e�s����������������������f�Z�L�������s�Z�K�=�E�`�s������������������俒�y�f�a�g�x���������ѿ�������ݿ���¦ª®®¦�������������!�-�S�����������l�!���-�,�.�:�A�S�_�l�x���������x�l�_�S�F�:�-�Y�Y�Y�Y�`�e�n�r�{�~�����~�|�r�e�Y�Y�Y�Y��߿ݿֿܿݿ޿����������������l�b�_�X�X�_�k�l�x�{�������x�l�l�l�l�l�lĚēďčĄĉĐĚĠĦĳĿ��������ĿĴĦĚ�Ľ������������Ľݾ��6�=�;�*����ݽн��)�#� �#�&�)�+�6�B�C�O�Q�O�K�D�B�6�3�)�)�#���
�����#�0�I�U�]�b�i�k�b�U�<�#���
����*�O�h�uƎƚƴƼƩƎ�u�\�*��#��
��������������
����#�-�/�4�/�#�`�^�T�I�L�T�`�d�m�q�y�������������y�m�`�ù¹����ùϹܹ������������ܹϹù��	���������	��"�'�/�6�;�=�G�;�/�"��	ìëåãìù����������ùìììììììì�/�)�"����"�/�8�;�H�T�a�g�h�b�U�H�;�/�H�E�@�E�H�U�a�n�o�s�n�b�a�U�H�H�H�H�H�HìçàÜ×ÓÒÐÓàìî÷ù������ýùì�������{�~������������������������������
�����$�0�6�=�=�=�;�1�0�$�����ùìéìïù������������������������������������������������������������ѾM�I�Y�o�������׾���	����׾�������M�/�(�$�(�(�*�/�;�H�T�a�m�u�y�~�w�m�T�;�/ìééìïùÿ����������ùùìììììì�������������������ʾվ޾���Ѿ��������m�`�T�N�G�D�G�Q�T�`�m�y�������������y�m��������������������������������������������ڻػػܻ������'�-�4�4�/������Y�O�M�Y�`�f�j�r�r�r�g�f�Y�Y�Y�Y�Y�Y�Y�Y�$� ��������$�)�-�.�$�$�$�$�$�$�$��������������	�������	���������r�\�L�E�B�C�M�f����������������������G�<�:�?�G�S�y���������ƽǽ��������y�`�G���߽�߽�����4�8�8�0�$������н��Y�Q�P�R�Y�e�r�������������������~�r�e�Y�����������������üʼּ�ݼ׼ּмʼɼ����*�$�#�*�2�6�B�C�O�[�^�^�\�X�T�O�N�C�6�*�V�N�I�@�F�I�U�V�b�o�{�ǈǈǈǅ�{�o�b�V��л��_�X�[�~�����л���@�Q�M�<�'����������������� �	��"�+�/�2�4�/�+�"��	����	��������� �'��������ā�t�s�v�{�āčĖĦĲĳ������ĿĳĦĚā¦¦²¿������������������¦�����#�#�/�:�<�?�H�R�K�H�G�<�/�#������������������������������������������D�D�D�D�D�D�D�D�D�D�D�EEE!EEEED�D��������������������������������������������������ùô÷ù��������������Ç�~�z�n�g�d�g�n�r�zÇÓØàãäàÔÓÇ�#��
���
��#�0�<�I�M�N�R�Y�^�P�<�0�#�ݿѿĿ¿Ŀѿݿ����5�N�l�s�Y�A�)����ݹ��������������ùϹܹ�������޹Ϲù�  D ? . 0 % Y 6 K A 3 a > J L f G  K O 4 e 7 ! 8 8 D  % \ 1 F ` M R A m - # "  8 = N  W G 6 ] c 0 j . H : & ? g U @ O " ( j D      �        �  �  �    �    ]  �  �  �  V  �  �  D     Y    <  }  '  q  �  �  �  5  �  %  '  �  K  �  �  J  �  �  �  Y  �  (  #  N  k    9  �  ;  �  }  �  �  �    R  �  f  m  �  Q  �  �  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@  D@    3  D  P  X  ]  [  R  B  *    �  �  }  H    �  �  L  
  �  �  v  X  7    �  �  �  n  >    �  �  |  ^  Z  �  �     ,  %         �  �  �  �  �  �  e  F  %  �  �  r     �   f    #  -  4  ;  :  6  $    �  �  �  �  e  J  /    �  �  �  U  K  B  9  0  &      	    �  �  �        :  [  }  �  l  �    /  Y  {  �  �  �  �  �  �  P    �  �  Q  �  �   �  �  �  �  �  �  �  �  �  �  �  �    y  y  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  o  c  }  �  �  t  b  I  ,    �  �  �  �  [    �  g  �  �   �   M  �  �      �  �  r    �  0  �  3  �  �  \    �  �  +   �  z  �  �  �  �  �  j  ;    �  �  i  "  �  r    }  �  A  A  >  ,      �  �  �  �  m  I  )    �  �  �  �  u  5  �  �  	/  	u  	�  	�  
  
#  
*  
  	�  	�  	M  �  w  �  I  �  �  f  Y   �  �  �  3  g  �  �  �  �  �  r  P  %  �  �  O  �  /  �    �      �  �  �  �  �  �  s  ^  K  9  '    �  �  m    �  R  �  �  �  �  �  �  �  �  �  �  s  b  Q  ?  .       �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  T  0    3  Z  |  �  �  �  �    4  =  3    �  �  d    �  :  �  �     �  r  �    ;  H  O  M  G  9  "  �  �  i  �  e  �  	  d  �   �  �  �  �  �  �  M  <  (    �  �  �  |  I    �  I  �  �  �  �  �  u  f  S  ?  +      �  �  �  �  e  <    �  �  �  ~  X  Q  E  6  !    �  �  �  a  Z  G  -    �  �  �    l     �  �  �  �  �  �  �  �  n  S  3    �  �  �  a  4     �   �  a  j  s  {  }  z  t  i  Z  E  (  �  �  �  I  	  �  �  4  �  �  �  �  �  �  �  p  W  9  ,  �  �  z  [  _    �  �  O  �         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �         7  Q  f  o  a  E    �  �  ~  8  �  �  �  �  H  �  �  �  �  �  �  �  �  �  �  �  �  ~  M  	  �    V  �   �  �  �  �  �  �  �  �  �  �  �    R  !  �  �  �  �  [  0      9  P  ^  c  [  G  %  �  �  x  M  .  �  �  ?  �    �  �  j  T  7    �  �  �  �  �  z  d  L  6       �  �  �  N    �  �  �  �  �  �  ^  +  �  �  x  2  �  �  M  �  �  _  $  �  �  �  �  �  �  z  i  R  ;  $  
  �  �  r  @    �  �  �  y  j  +  2  =  P  _  S  5    �  �  �  d    �  r      �  �  a  k  o  m  q  g  T  :    �  �  �  z  V  )  �  �  \  K  P  �  �  �  �  �  �  �  �  �  �  l  S  8    �  �  _    �  �    A  N  [  ^  X  S  L  <  #    �  �  �  G    �  6  h   X  Q  P  R  Z  b  `  M  4    �  �  �  q  A    �  r     �  �  �  �  �  �  �  �  �  n  [  D  ,    �  �  �  �  S    �  U    k  {  {  p  Z  6  �  �  G  �  �  O      �  �         �  �    }  {  x  s  n  j  g  b  [  U  M  E  ?  9  T  �  �  �  �  �  �  �  �  �  h  N  4    �  �  �  �  i  =    �  !    ;  O  P  P  P  N  K  D  :  '    �  �  �  �  i  ?    �  r  �  �  �  �  �  �  �  m  #  �  s    �  @  �  �  >  �    �  �            �  �  �  �  �  �  x  8  �  �  G  2   �  �  �  �  �  �  �  �  k  S  0    �  �  �  P  -    �  �  L  �  
  *  A  T  \  [  M  :  $    �  �  _    �  K  �  �  (  �  �  �  {  i  T  <  #    �  �  �  ]  /    �  �         �  �  �  �  w  Y  ;    �  �  �  �  q  ?  �  �  ~  \  w    �  �  �  s  [  @    �  �  �  w  >  �  �  ?  �  n  �  �   �  e  V  <    �  �  �  [    �  �  k  <  (  �  �  v    �  F  �  �  �  �  �  �  x  ^  B  !  �  �  �  e  -  �  �  d    �    �  �  �  m  F  +      �  �  �  C    �  ~  >    �  �  �  �  �  �  �  �  �  |  U  $  �  �  �  t  >  �  �  Y  �  �    A  A  9  ,    �  �  �  W    �  c  �  �     R  �  �  8        �  �  �  �  �  W  (  �  �  x  +  �  �  -  �  a   �  �    '  ;  @  >  7  *      �  �  �  \    �  i  !  �   �  �  �  �  �  �  g  A    �  �  �  �  �  �  �  �  {  q  �  �  h  �  �  �  �  �  z  d  E    �  K  �  �    �  M  �  c    �  �  m  O  Q  N  -      )  #    �  �  �  p    �  �  �  
d  
X  
A  
   	�  	�  	�  	D  �  �  F  �  w  	  �    X  �  S    )  �  �  �  �  r  X  6    �  �  �  e  "  �  h  �  "  <  �  �  ,  &      �  �  �  z  5  �  �  B  �  �    �  �  d  :  ^  J  3    �  �  �  �  �  �  j  >    �  �  o  <  	  �  �