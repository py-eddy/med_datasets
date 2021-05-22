CDF       
      obs    H   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��t�j~�        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       MݼX   max       P�v�        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��-   max       <���        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Y�����   max       @E��Q�     @  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v}�����     @  ,L   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @3         max       @Q�           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�8@            8   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���`   max       <�9X        9<   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�t�   max       B-NO        :\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�]�   max       B-��        ;|   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >G�g   max       C���        <�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >F�z   max       C��Y        =�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          `        >�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C        ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          C        A   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       MݼX   max       P�v�        B<   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Fs����   max       ?�A��s        C\   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��9X   max       <���        D|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�\   max       @E�p��
>     @  E�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v|�����     @  P�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q�           �  \   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�8@            \�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B)   max         B)        ]�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�|����?   max       ?�>�6z�     �  ^�                  #                  0      `         	                     )                        
            (      '             $      B   (                           8         
                                          
   N �N_�uNzE�PJNe�;O��N�9AN���O,WO�N�N���P,�2O�܎P��MݼXNN�Ny��Oɝ�N�`NN�N��kP�rO�gIO�}�N�3NOMO��KN*��N:��O]
~N�E�N��N �LP=��Nc�DP.�KN:�PF��P��N��4OIIOk�OdP�v�P#m�O�;N��iOE(LO9�MOKC`O�1Ou�OD/&PKYzN��N���N���O�O��2NWNIO�N,N;$NǺ1O'��NJ�)O2�DO�N�hAN��Ng{PN�3�<���<�9X<���<e`B;ě�;D��;o:�o��o��o��o��o���
���
�ě��o�o�t��#�
�D���D���u��C���C���C���t���1��1��9X��9X��9X��9X��9X��9X�ě��ě����ͼ��ͼ�������������`B��h�������+�+�C���P�����,1�,1�,1�0 Ž0 Ž0 Ž0 ŽH�9�P�`�P�`�Y��]/��o��O߽�O߽��P���-���{��-���

�������������)6@76+)'gmzz{|{zmkdcgggggggg��������������������HO[hht{th][WOLHHHHHH����
*/79:2/#
������������������������@BMN[_`a[NB>@@@@@@@@GN[gt����{tgc[NNKFGGnrtt������������tgn��� �����)<BO[`cca^WB6)���BOXXNB)�����#<Hz������~a/#���

���������������������������������"##/5;<=<3/#����������������������������������������~�����������|{~~~~~~���������������������������������������HTajmprz�����zTHC?@H�������#%!�������DO[bhotxtnh[OJFDDDDD��������������������=BOZht������}�t[O>;=NOOW[bhlhe[ONNNNNNNNST[aaa_ZTKKMSSSSSSSS`fmz���������|ja\\\`��������������������������������?BKNRQONB=??????????�
#>Pbn{��vkhJ<0������� ����������������)O[htyyvh[6�����55BNNO[[[SNB85555555�����
"-/)��������Ycx��������������g\Y���������������������������������������� *5BNY[dgjf[NB1)HILOTUagnnppnngaUKHHr�����������������rr���������������~�{|�}����������������}}}�������������������������

	����������������������������������
",/(#
�����38NSgt���������VN503����������������������������������������NWgt���������tg[NIGN!#+0<<A?C><20,##!!!!+/6<DHIIIHH<<5/,'$++EHUX^^]UHA<>EEEEEEEEKUamz�����znlaZURNK���������������������������������������������������������))**+)("##,/<?=<0/-#########������ ������������

�����X[dgjqngg[YSXXXXXXXXx��������������vutx��������������������//049;<HJMQSPIHC<///����������������������������������� 	"	�����z�y�v�x�z�ÇËÒÇ�z�z�z�z�z�z�z�z�z�z���ֺкҺкֺ�������������ÓÊÏÓàìù��ûùìàÓÓÓÓÓÓÓÓ�/�%��������������/�L�W�s�{�s�a�U�H�<�/�z�n�w�z�zÅÇÈÐÓÔÓÑÇ�z�z�z�z�z�z������ŹŬŮŹž����������������������z�q�z�z�������������������������z�z�z�zƎƍƁ��wƁƎƚƦƥƟƚƎƎƎƎƎƎƎƎ��������(�.�4�;�=�A�I�A�@�4�(���A�5�(�������������(�5�A�L�M�U�U�O�AàÜÓÊÇÅÂÇÓÔàììùûùùìàà�T�;�.����پ�	�.�G�y�������������m�T�Z�S�L�O�W�Y�W�Z�g�s������������}�s�g�Z������»»�����
�#�<�H�U�X�d�c�H�#�
���˼Y�V�U�T�Y�f�n�f�f�Z�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y�¡�6�.�)�)�)�3�6�B�F�O�S�O�N�J�B�A�6�6�6�6��������������*�C�O�V�\�_�^�X�O�C�6����Z�T�Q�Z�g�p�s�u��������s�g�Z�Z�Z�Z�Z�Z�;�2�/�"�"�"�/�;�H�R�T�]�T�H�;�;�;�;�;�;�нƽĽ����Ľнݽ����������ݽннно4�*�!�$�4�K�s�����ľ���ʾ����s�Z�A�4ƿƩƠƧƭ������� ���!�$�!��������ƿ�$������� �0�=�I�V�[�\�W�I�>�5�0�$�Y�M�T�Y�^�e�n�r�v�~�������~�r�e�Y�Y�Y�Y�ʼ¼ʼʼּ������ּʼʼʼʼʼʼʼʻ_�X�X�l�z��}�������ƻ��ûллɻɻ��x�_�Ϲƹù��������ù˹ϹعйϹϹϹϹϹϹϹ��g�e�g�p�s�u�����������s�g�g�g�g�g�g�g�g������������!�(�5�A�N�Y�N�A�5����ݿѿοƿѿݿ������������������������������������������������������
��������
���!��
�
�
�
�
�
�
�
�
�
�����}�g�Z�J�K�V�g�s�������������������������������������������������������������s�b�[�W�n�k�q�q�����������������������s�������y�y�y�����������������������������m�n�������ּ���!�.�;�7�"����ּ������m���m�T�G�8�,�+�;�a�z�������������������������������������������������������������L�G�E�@�E�L�e�r�~�������������z�r�e�Y�L����������������)�7�?�=�6�0�)��s�g�Z�N�A�>�:�A�L�N�Z�g�s�u���������t�s�_�M�:���ں޻�����ܻ�� ����ܻ������l�_���3�Y�r���������ɺҺٺɺ����Y�L�@�3��5�+�)�������)�5�B�N�Q�V�N�K�B�5�5����ùìëáàßàéìù���������������Ž��������������������Ľнֽ۽ݽܽнĽ����U�S�H�L�a�n�z�}ÁÃÇËÓÓÇ�~�z�n�a�U�$�������� ���$�0�=�B�H�H�C�=�2�0�$����ݾ������	����"�.�2�"���	�����	������	��"�.�0�;�@�;�9�.�-�"��ùöóù�����������&�����������ù¦�v�s�t�r²�������	��������¦�_�\�S�G�Q�S�_�l�x���������x�m�l�_�_�_�_E�E�E�FFFFF$F1F=FAF>F=F1F+F$FFE�E��/�+�,�/�;�H�T�_�`�T�H�;�/�/�/�/�/�/�/�/�����������ùϹӹܹ������ܹչϹù��`�T�;�/�-�1�1�.�'�.�;�G�R�f�t������y�`�U�J�M�U�a�b�n�r�u�n�g�b�U�U�U�U�U�U�U�U�0�,�$���$�0�=�I�V�[�b�c�d�b�V�I�=�0�0�s�p�f�Z�V�M�J�M�Z�f�n�s�����������s�s�ĿĿ��������ĿſѿѿѿſĿĿĿĿĿĿĿĺ�������������!�"�!������������ED�D�D�D�D�EEEE*ECEPEXE]EPE7E+EEE����������������������������������������ŭŠŤŠŔőŜŠŭŹ����������������Źŭ���������ĿͿѿؿҿѿǿĿ���������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����������������������������������������̽!������!�.�4�9�.�%�!�!�!�!�!�!�!�!��r�f�^�Y�R�M�A�M�V�Y�f�r�u���������� T a X m b ) W 9 G @ Z P p & v 7 ^ _ 4 _ N z l r 4 U F @ \ \ U 4 t m ? h U a 7 > T 3 E g U J T ; I , W + r / B Y , R � L C v S - { C 4 / N ] ) �  3  �  �  E  �  �  �  �  U  x  �  J  !  q  A  6  �  ,  �  s  �  �  `      p    M  q  -  �  �    $  �  �  x     �  �    �  ;    F  3  #  �  �  �  �  =  -  �  �    �  [  E  k  ?  �  6  �  �  R  �  )    �  s  <�9X<u<e`B�e`B;o��󶻃o��o��`B��9X��t��T���ě����ͼo�u��C��\)�u��o���ͽ�P�#�
�ixս'�9X�@��o��/�\)��/����`B����`B��o��h��%�ixս+��㽁%�<j�ě���O߽�P�H�9�P�`�aG��e`B����P�`��\)�ȴ9�<j�aG��Y��m�h�}�]/��%�m�h�e`B�y�#��j��t�����ȴ9��-�� Ž�����`B1tB�A��B�B^]BoB��Bc�B	"KB
�RBgdBB%�BnuB#�B��B�B��B�B)PB)�oB!L�A��PB�mB7�B �(B�B;�A�t�A�ѫBd�BMB�B%ޖB̂BP@B�DB-NOB('B'�B"B��B�B�mBɨB
�nBatB#(�B!�rB�NB	;jB!�B��B
 CB%�lB�?ByCB��B,�B8�BQ�B��BN�B#:�B��B	GB
�nB��B�/B=�B�B�QB=PB�yA���B�TB�\B4	B��BTUB	qB
��BBB��B@.B��B$;�B=VB��BA�B8 B�B)��B!@3A�]�BԵB?�B C�B�CBA(A�xdA���B�2B�gB�B&.�B�qB=SB;�B-��B
�\BC�B"1�B�cB?�BB�PB�B�*B"ɨB!� B:/B	EB>B��B
=)B%�9BTDBAFBV^B+�B?�B?B��BB2B#!�B��B	mB<B��B�B�B�^B�\A�S@?R�A��A��uA��
A�NA���B��A7VA��A˘�Ae�eA���A�A@ڛ�A��TA�YA���A��!A��_A*�BAC��BA�B
�&?�cA��@���>G�gA�!�A��A~U A�A�PA��~A�:�A�p�AoIxA �#A�%�A�O�?��<AՆ�A�#�@�/@�DA��A�aA%a�Aǖ�B	�0AZS,A^�A���A���@�
�C���A�D�>���Ai+BA�[DB
�~AC<9AxF�@\hMC���A��A��)Aw8qC�lA�zA-@��0Aȝ#@C�.A�3+A�k�AȨA��A�}@B�JA7nA�[AˀAe=�A���A�Un@�#�A��AיAA��A�a|A�m�A+x�A@��B�<B
��?��6@���@�,>F�zA�-�A��A~zA�V;A�i�A��A�U�A�p�Ao�A�A�B�A�b�?�:A�]�A��8@�v@�A��A�s�A&�NA�uB	�0AZ��A^��Aϙ�A�~�@���C��YA�ix>�DmAeA�A�{�B8�ACVAx�0@[�!C��3A�H�A��=Av��C�tA�^�Ai@�                  #                  1      `         
                     )                        
            )      '             $      C   (                           9                                                   
               +                        /      9            %            5   '   !         )                     5      /      ;   +               C   /                  !         /               %                                                   %                        %      -            !            '   #            #                     5      /      9   +               C   -                           )               %                                       N �N_�uNzE�O�EJNe�;OmfN F�N���N�O�N�N��#OŦ�N�%�PI��MݼXNN�Ny��O�պN�`NN�N���Oؗ�O�hO$-�N�3NOMO�i
N*��N:��O]
~N�E�N��N �LP=��Nc�DP*|N:�P@�P��NEB&O.Q�Ok�OdP�v�P��O�;N���OE(LO9�MOKC`N�H�N�&�O$;�P�HN��N��8N���O�O��2NWNIO�N8�*N;$NǺ1O'��NJ�)OG4N��$N�hAN��Ng{PNc~g  �  �  �  �  u  )  �  �  X  5    ;  E  	�  �  y  2  K  #  �  i  :  �  �  �  �  �  �  ,  �  �  +  �  �  h  �  R  $    �    H  �  s  �  M  �  )  �  �  B  V  9  S  �  N    �  �  H  �  �  �  �  �  �  "  N    �  �  �<���<�9X<���<t�;ě����
$�  :�o�o��o�ě���t��u��`B�ě��o�o�D���#�
�D���u���㼓t�����C���t���9X��1��9X��9X��9X��9X��9X��9X�ě����ͼ��ͼ���������`B��/��`B��h�������\)�+�C���P�L�ͽ�w�49X�T���,1�49X�0 Ž0 Ž0 ŽH�9�P�`�T���Y��]/��o��O߽�hs���㽝�-���{��9X���

�������������)6@76+)'gmzz{|{zmkdcgggggggg��������������������HO[hht{th][WOLHHHHHH���
#.353/)#
�������������������������@BMN[_`a[NB>@@@@@@@@KN[gt{���ztga[QNNIKKnrtt������������tgn�����������#26BOW[[[XOB60#!)5BEMHB853)##/Un������}aU;##���

���������������������������������"##/5;<=<3/#����������������������������������������~�����������|{~~~~~~����������������������������������������HTahmoquz����zTHC@@H������������DO[bhotxtnh[OJFDDDDD��������������������>BOYht������}t[O?>;>NOOW[bhlhe[ONNNNNNNNST[aaa_ZTKKMSSSSSSSS`fmz���������|ja\\\`��������������������������������?BKNRQONB=??????????�
#>Pbn{��vkhJ<0������� ����������������)O[txyuh[6�����55BNNO[[[SNB85555555��������	",.'�����Ycx��������������g\Y���������������������������������������� *5BNY[dgjf[NB1)HILOTUagnnppnngaUKHHr�����������������rr���������������~�|}�}����������������}}}�������������������������

	����������������������������������
",/(#
�����KN[gt�����tg[WNJKKKK����������������������������������������OTT[gt���������g[NLO!#+0<<A?C><20,##!!!!-/2<BHHIHG<6/-(%----EHUX^^]UHA<>EEEEEEEEKUamz�����znlaZURNK���������������������������������������������������������)*)&!##,/<?=<0/-#########������ ������������

�����X[dgjqngg[YSXXXXXXXXz��������������{wvzz��������������������//049;<HJMQSPIHC<///������������������������������������
��������z�y�v�x�z�ÇËÒÇ�z�z�z�z�z�z�z�z�z�z���ֺкҺкֺ�������������ÓÊÏÓàìù��ûùìàÓÓÓÓÓÓÓÓ�/����������������#�/�C�U�c�r�i�U�H�<�/�z�n�w�z�zÅÇÈÐÓÔÓÑÇ�z�z�z�z�z�z����ŻŴŴź�����������������������������z�x�z�������������������z�z�z�z�z�z�z�zƎƍƁ��wƁƎƚƦƥƟƚƎƎƎƎƎƎƎƎ��������(�,�4�:�;�A�G�A�4�4�(���A�5�(�������������(�5�A�L�M�U�U�O�AàßÓÌÈÓàéìùùùðìàààààà�;�.���	��"�;�T�a�y���������y�m�T�G�;�s�r�g�c�a�e�g�s���������������t�s�s�s�s�������������������
�#�8�H�U�V�H��
���ռY�V�U�T�Y�f�n�f�f�Z�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y�¡�6�.�)�)�)�3�6�B�F�O�S�O�N�J�B�A�6�6�6�6�����������������*�6�C�X�U�O�C�6�*���Z�T�Q�Z�g�p�s�u��������s�g�Z�Z�Z�Z�Z�Z�;�2�/�"�"�"�/�;�H�R�T�]�T�H�;�;�;�;�;�;�Ľ��ýĽнݽ���������ݽнĽĽĽĽĽľA�9�3�)�.�<�R�s�������ʾľ������n�Z�M�A��ƪƢƧƱ������������#� �����������0�(�$� ��!�$�,�0�=�I�R�U�V�M�I�?�=�0�0�Y�M�T�Y�^�e�n�r�v�~�������~�r�e�Y�Y�Y�Y�ʼ¼ʼʼּ������ּʼʼʼʼʼʼʼʻ_�Y�Z�l�{����������û����ϻȻȻ����x�_�Ϲƹù��������ù˹ϹعйϹϹϹϹϹϹϹ��g�e�g�p�s�u�����������s�g�g�g�g�g�g�g�g������������!�(�5�A�N�Y�N�A�5����ݿѿοƿѿݿ������������������������������������������������������
��������
���!��
�
�
�
�
�
�
�
�
�
�����}�g�Z�J�K�V�g�s�������������������������������������������������������������s�c�\�X�o�l�s�������������������������s�������y�y�y����������������������������������r�q�������ּ���!�.�:�6�!����ּ����m�T�G�8�,�+�;�a�z�������������������������������������������������������������Y�S�L�I�F�B�G�L�e�r�~���������}�x�r�e�Y����������������)�7�?�=�6�0�)��s�g�Z�N�A�>�:�A�L�N�Z�g�s�u���������t�s�_�M�:���ں޻�����ܻ�� ����ܻ������l�_�!�!�+�G�Y�r�������ͺֺѺȺ����Y�L�@�3�!�5�+�)�������)�5�B�N�Q�V�N�K�B�5�5ùîìãáëìùü����������ùùùùùù���������������������Ľнֽ۽ݽܽнĽ����U�S�H�L�a�n�z�}ÁÃÇËÓÓÇ�~�z�n�a�U�$�������� ���$�0�=�B�H�H�C�=�2�0�$�������������	��������	�����������	�� ���	��"�-�.�;�>�;�7�.�"�!��ù÷õù������������������������ù¿²¦�~¦²��������� �����¿�_�\�S�G�Q�S�_�l�x���������x�m�l�_�_�_�_E�E�E�FFFFF$F,F1F<F1F*F$FFE�E�E�E��/�+�,�/�;�H�T�_�`�T�H�;�/�/�/�/�/�/�/�/�����������ùϹӹܹ������ܹչϹù��`�T�;�/�-�1�1�.�'�.�;�G�R�f�t������y�`�U�J�M�U�a�b�n�r�u�n�g�b�U�U�U�U�U�U�U�U�0�,�$���$�0�=�I�V�[�b�c�d�b�V�I�=�0�0�s�r�f�Z�Z�Z�a�f�i�s�����������s�s�s�s�ĿĿ��������ĿſѿѿѿſĿĿĿĿĿĿĿĺ�������������!�"�!������������ED�D�D�D�D�EEEE*ECEPEXE]EPE7E+EEE����������������������������������������ŭŤťŠŠŞŠŭŹ����������������Źŭŭ�������������������ĿȿѿֿѿпƿĿ�����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����������������������������������������̽!������!�.�4�9�.�%�!�!�!�!�!�!�!�!��w�r�f�_�Y�S�Y�f�r�u����������� T a X h b  ^ 9 H @ P < 2 3 v 7 ^ ] 4 _ N f m 8 4 U H @ \ \ U 4 t m ? k U a 7 @ 9 3 E g U J A ; I , J ' g . B L , R � L C h S - { C , , N ] ) l  3  �  �  ^  �  �  g  �  #  x  �  �  �  i  A  6  �  �  �  s  �  �    p    p  �  M  q  -  �  �    $  �  �  x    �  m  �  �  ;    ,  3  �  �  �  �  8    �  �  �  �  �  [  E  k  ?  s  6  �  �  R  <  �    �  s  �  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  B)  �  �  �  �  �  �  �                �  �  �  �  �  �  �  �  �  �  �  �  m  Y  E  F  L  N  =  -      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  a  P  @  1  !    �  �  �  �  �  �  �  �  �  �  �  �  h  /  �  �  ?  �    5  u  p  l  g  c  ]  Q  E  9  -      �  �  �  �  �  �  �  �  �  
    #  )  %      �  �  �  Y    �  y    �    �  7  �  �  �  �  �  �  �  �  �  �  {  m  _  J  '    �  �  �  v  �  �  �  �    q  c  V  H  :  ,        	         �  �  T  U  V  W  T  Q  M  F  ?  7  +    
  �  �  �  �  �  �  }  5  3  -  %        �  �  �  �  f  -  �  �  f  +  �  �  �  �  �  �    �  �  �  �  �  ^  :    �  �  �  z  9  �  �  h  �  �  �    $  7  :  2  '    �  �  �  �  N  �  y  �  �  @  !  %  !      �  �  �  )  =  E  3    �  �  �  W    �  �  �  �  �  	W  	  	�  	o  	?  	  �  �  f  �  t  �  #  =  d  g  d  �  �  �  �  �  v  j  ]  Q  E  ?  >  =  <  ;  :  9  8  7  6  y  k  ^  N  ;  '    �  �  �  �  �  `  #  �  �  �  k  <    2    
      �  �  �  �  �  �  c  C  !  �  �  �  �  _  4  �  +  J  F  ;  )  	  �  �  v  C    �  �  w  !  �  [  �  t  #  "  !               �  �  �  �  �  �  �  �  �  �  j  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  U  [  a  h  i  e  _  O  7    �  �  �  s  F    �  �  �  E      &  9  6  +    	  �  �  �  �  �  �  s  C    �  h    �  �  �  �  �  �  �  �  �  �  �  �  �  S  �  r  �  %  z   �  �  �  ~  �  �  �  �  �  �  �    M    �  �  -  �  K  �  ?  �  }  c  E  '  �  �  �  �  ~  Q    �  w    �  �  M  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  s  �  �  �    9  �  �  �  �  �  �  q  Z  ;        �  �  T  �  r  �  �  O  D  �          �  �  �  �  �  �  �  �  j  R  9    �  �  �  �  ,    �  �  �  �  �  e  F  %    �  �  �  |  Z  /     �   �  �  �  �  y  i  W  L  l  k  Y  ?  !    �  �  �  T    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  i  [  M  ?  +  $                           �  �  �  �  �  �  �  �  �  �  �  $  L  W  T  Q  K  D  <  0      �  �  �  |  �  {  V  -    �  �  �  �  }  z  �  �  �  �    h  J     �  h  c  ^  Y  T  O  K  H  D  @  B  K  T  ]  f  a  X  N  D  ;  �  �  �  �  �  �  l  ?    �  r  "  �  �  Q  J  �  �  Y     R  B  1  !       �  �  �  �  �  �  s  Z  A  /       �   �  "    �  �  �  �  �  }  p  h  Y  ;    �  �  l    �  �  f    �  �  �  �  �  �  �  �  X  .    �  �  u  D  #  �  �  !  �  �  �  �  �  �  �  �  �  �  �  �  �  o  Y  =  "  �  �  �  �  
    �  �  �  �  �  �  {  n  m  ^  N  ;  &    �  �  d  H  A  ;  3  &    �  �  �  w  :  �  �  l    �  >  �  �  �  �  �  �  �  �  �  �  �  l  N  .  
  �  �  ~  D  �  O  �  S  s  X  '  �  �  8    >    �  �  )  �  [  +    �  U  ~  >  �  �  �  ~  p  l  b  A    �  �  E  �  �  j    j  �  >   �  M  F  @  9  5  1  -  *  '  $         
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  F    �  �  �  \  *  �  �  �  )  %        �  �  �  �  �  m  L  '  �  �  �  <  �  �  b  �  �  l  D     �  �  �  }  Z  >    �  �  x  ,  �  �  �    �  �  i  C    �  �  �  �  q  F    �  �  �  �  o  B    �  �  �  �      $  *  1  ?  =  %    �  �  c    �  
  k   �  Q  U  T  O  D  :  0  #        �  �  �  �  j  8  �  �  &  �  �  &    �  �  m  4    �  �  �  b  $  �  �  C  �  D  �  )  B  O  P  Q  ?    �  �  s  5    �  �  y  3  �  �  �  �  �  �  �  |  w  p  d  Y  M  B  :  6  3  /  +  #         �  =  H  B  %    �  �  �  s  N  (    �  �  \     �  �  �  R    �  �  �  �  �  �  h  I  &    �  �  �  �  }  M    �  g  �  �  �  �  �  m  Z  C  )    �  �  �  l  ,  �  �  A    �  �  �  �  �  �  �  �  �  �  g  D      �  �  �  s  =    �  H  B  <  6  0  &        �  �  �  �  �  n  P  /    �  �  �  �  �  �  �  r  _  I  2    �  �  �  �  S    �  \     �  �  �  �  �  �  �  �  �  �  �  b  1    �  �  �  w  O  &   �  �  �  �  �  �  w  e  \  X  S  O  J  F  <  )      �  �  �  �  �  �  �  �  y  m  b  V  H  :  *    �  �  �  �  �  �  �  �  �  �  �  �  X    �  �  �    !  �  �  �  ^    �  �  5  �  �  |  v  p  j  d  \  S  J  B  9  0  &        �  �  �                 �  �  �  �  �  }  [  3    �  �  �  �  2  A  N  M  A  /    �  �  �  �  \  '  �  �  m  '  �  �  T    �  �  �  �  �  s  U  8       �  �  �  �  �  z  d  R  @  �  �  �  ~  |  x  o  e  [  R  E  6  '      �  �  �  �  �  �  �  �  �  �  ~  t  g  X  H  6  "    �  �  �  �  �  _  +  �  �  �  �  }  A    �    ;  �  �  �  I  �  ~    �  =  �