CDF       
      obs    I   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��$�/     $  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��(   max       P�"�     $  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <T��     $  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?@        max       @F(�\)     h  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �У�
=p    max       @v{��Q�     h  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @O            �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @�*�         $  8|   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��l�   max       :�o     $  9�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��6   max       B0Tu     $  :�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�a^   max       B0=n     $  ;�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >� s   max       C���     $  =   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��,   max       C���     $  >0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          T     $  ?T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =     $  @x   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9     $  A�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��(   max       P�ԝ     $  B�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��3���   max       ?�����m     $  C�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��^5   max       <T��     $  E   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?@        max       @F(�\)     h  F,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @v{��Q�     h  Q�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @K@           �  \�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @�I�         $  ]�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Bl   max         Bl     $  ^�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n   max       ?�����m     �  _�      *   
   (                  *   !                  	            0      #   ,      G   .            *                    T                     F                                 
      )      
            "                     (   	      !O�O�j�OUCFPX\]O �rN�@(N>��NI�N>�O�},Ou�!NCM�N~��OoK�O,�NNa�jN�TN�RN�U�N�2NOƳ�N�O�P�O�2oO�LP�=P
j1Om�'NJ�N�t5P:�\O~ӆO��N��?Ok��O�-�P�"�OA�3N�.Oj�NÖ�Nc�O$75P�O7�nN<��O��&Nn�$Nj�#M��(N��BN�O�z�N�cN��0M�'O��Oy-N�0�O9^AO��Oo�P��N�+�N�	�Og�N�EO�O�OF��O��Nx�*Ow��O
v�<T��<o<o;ě�;�o;�o:�o%   ��o�o��o���
�o�D���D���T���u�u��o��t���t���t���1��1��9X��9X��9X�ě��ě����ͼ��ͼ���������/��h�������������+�C��C��C��C��C��\)�t���P��P�����#�
�#�
�,1�,1�0 Ž@��@��@��@��@��D���P�`��%��o��o��t����������P��������������������������
#/=MSUPD/#
������������������������3[k���������l[D:<573������������������������������������������������������������_afmnrtmaa[[________8<FIUW_ZUIB<88888888��
#%%%
�������������������������knnz����znkkkkkkkkkk������������������������������������������������������������36BDDHCB65*/33333333?BN[]dgjgf[NJB>9????7BO[chmlh_[OMCB=7777���

��������#$/4<><?=<2/*#��������������������&*/6CFLOOPOMC6*%&&amz���������zmaXUUYa��������������������mqz����������zmhbdgm;DUan�������saUH>:8;6B[h���th[OF6�KTamz������zma^XSQOK�����������������������������tst��������6<Haz�������zmaTD:56gt�������������tg^]g��������������������dhnst~���������theddOht������}mh[OHHHDDO#)BNTV[^`dd[B5)%BKcet���������tN0+-B���������������������������������������������������������������������������������������������������

�������������������������������������������������������������������_k{������������{nb]_��������������������,/:<?BB</*'-,,,,,,,,$fgjot�������tkggffffX[gtuwtg[QXXXXXXXXXX#<UijbZSD<0*������������������������������������������������������������+4<IUbnuxzwnbUI<1,*+qt������������tmjkq��������������������������
������������)6AA7)���������������������������� �����������wz{��������������zxw��"$������	
#*//671/+# 
	����������������������"'*-,)���������������������t��������������tpkntyz���������}zzyyyyyy����������������������������������������Z�P�N�A�?�N�Z�g�s����������������s�g�Z����űŪũŨŭŹ��������� � �����������ƿa�T�R�J�G�F�G�I�Z�`�m�y�������������y�a�A�)��������(�5�N�q�����������s�g�Z�A�����������
���#�/�4�1�/�'�#���
������ؼּ˼ּؼ�����������������}������������������������������t�p�t�āčĚĤĚĚčā�t�t�t�t�t�t�t�t�ʼü��������ʼּڼּܼѼʼʼʼʼʼʼʼ��N�A�1�-�5�A�N�Z�g���������������s�g�Z�NĚčā�~ĈčĚĦĳĿ��������������ĿĳĚ���������������������������������������޻�ݻܻһٻܻ�������������������������������������������	��	���������ɽнɽ��������Ľнݽ�������	�������пT�N�J�T�`�m�y�{�z�y�m�`�T�T�T�T�T�T�T�T�y�q�p�n�y�����������������������y�y�y�y�e�Y�Z�_�e�l�r�~�����������~�}�r�e�e�e�e��߽ݽ׽ݽ�������� ����������H�<�<�1�/�)�/�<�H�U�U�a�c�a�_�U�H�H�H�H�F�4�*����'�@�f�r������������q�f�Y�F�����������	����"�&�)�&�"��	����ƱƩƥƥƢƧƳ���������������������Ʊ��������������������$�0�>�=�6�0�$����������������#�)�0�<�G�E�G�A�<�0�#���������������������ùܹ�������ܹù��]�a�^�g�o�p�m�����������������������s�]�����������������	��� �"���	��������ƎƆƁ�}ƁƎƚƤƣƚƎƎƎƎƎƎƎƎƎƎ���������������������������������������������ݿͿʿѿݿ����5�Q�g�p�l�N�5�����������������������&�$��������ɾľ������ʾ׾����.�'�	�����޾۾׾��<�6�/�#�!�����#�/�<�>�H�U�T�J�H�<�<�k�p�n�o�w�x�~�����������������������x�kÇ�z�t�c�a�n�zÄÓàìùýÿþûùìàÇ����¦¦��������L�V�P�a�/��
��~�z�r�e�`�b�l�r�����������������������~ŹŸŭŠśŔőŔŠŭŹ��������������ŹŹ�x�o�l�_�W�S�F�>�:�F�S�_�l�x�����������x����������!�-�0�:�-�!�����������������	��	��"�%�$�"�����������g�^�[�N�N�M�[�g�t¦�t�g�S�:�.�"�&�-�4�S�l�x�����������������x�S�ֺԺɺĺɺͺ����������������FFFFFFF$F1F4F4F1F$FFFFFFFF�(����޿׿ٿ�������A�K�T�T�H�5�(�;�:�0�;�A�H�T�U�a�k�a�T�H�@�;�;�;�;�;�;�O�I�O�O�[�h�tĀ�w�t�h�[�O�O�O�O�O�O�O�O������������������������������������������ƹƳƧƞƞƧƳ�����������������������������������	����	���������������������лû������Ļл�����������ܻ��)�"�"�$�)�*�6�B�H�M�O�V�O�F�B�6�)�)�)�)����
������'�)�6�8�6�)�(�������ּҼԼּ�������������㽞�������������������нؽ����νĽ����������������*�C�O�Y�U�W�O�C�6�������������ż�����������������������àßÖÏÈÁÇÉÓàéìõý����üùìà��f�Y�R�L�K�N�W�c�r�������������������'�������'�-�4�@�M�Y�\�Y�T�M�@�4�'�f�`�����ʼ���������ּ�������r�f��������������������������������ŇŇņŇŇŌŔŠŭůŮŭšŠŗŔŇŇŇŇE*E*E'E%E*E6E7ECEPE\EaEiEpEiE]E\EPECE7E*�H�B�A�H�L�T�^�a�m�u�x�m�l�a�V�T�H�H�H�H�l�g�N�G�@�G�K�S�`�l�������������������l�����������������
���"�!���
��������ĹįĨĬĳĿ���������
�#�(�$��������Ĺ�׾Ӿʾ��ʾ׾�����������׾׾׾׾׾��S�I�B�8�0�$��$�0�I�V�b�o�}ǁǄǁ�{�o�S�a�f�h�a�\�T�H�F�;�3�/�*�'�/�;�H�T�a�a�a Z 2 7 I R # J b J E S ] 2 A 2 I 4 @ / D < " 0 M " - a : 2 | V F [ R C * N ! Q _ A G 6 1 ) m _ S P Z h c 2 3 H Q   j U C H i U d 6 ` > > @ Y W ,    q  �  �  �  /  �  u  �  h  N  1  �  y  �  z  p      y  �  �    r  �    `  �  �  g  �  �  &  y      I  �  �    +  �  0  m  �  �  �  �  �  �  "  �  P  �    �  ,  f  �    �  u  D  X  5  �  A  �  <  �  �  �  ,  +:�o�C���o�+��`B�o��o�D����`B�0 Ž�P�#�
�u���ͼ�/���㼼j�C���9X����+���aG������w��j��C��0 ż�`B�o��+�,1�,1�'y�#�}��l��49X�t��H�9��w�\)��o�����8Q�#�
�e`B���0 Ž�w�,1�#�
�q���P�`�H�9�8Q콩��ixսixս�\)���-�]/���T����q����j��\)��Q콾vɽ�l����T�����"�B{AB�rB l;B
~�B�CBI�B(#A��6B&�%B��BB\�B9�B�B!�7B��Bh�B�eB�KB*BB" QB0TuA��B=B �B��B�@A�h�B�lB>oA�n�B
�iB��B�B�cB�_B	��B�NB
�8B!�B"��B,�B\�B�TByBȁB)0RB��B��B�xB	��B	-.B%��Bm�Bh�B �B'5�B
��B��B�DB*B*B-$B@B�ZB�Bn�BHABP�B
��BӃB�XB��B>�B��B �(B4BB�B@B�A���B&��B�=B?�B��B=�BE�B!��B� BQ�B�BD|B>�B!��B0=nA���B��B �B��B�	A��BAQB�"A�a^B
��BȇBVxB��B��B	O�B�0B#�B"#B#?�B�B@�B�LB@HB�B)?�B�,BKbB3MB	�B	?�B%~BJB@*B ҉B'A,B
�B@%B�B�yB)��B,��B@qB�BB��B@ B�jB
��B�~B�
B�+A�A��WAj�]A��A�>;Al�@�|(A�@��A�j�A�=�A�4d@�z�A�AA+TAi��Ao��?���A-��A��d@ؒ�A[�HB�}B	&�A�(>� sA�x�A��B�A�bBA�'A�UQAYCA���@���A��EA��@��A�G�@�Ls@c��A\�MA��h@�=g@L�\C���A��xA��hAے�A��`B�=AZX@���AטA�w�A�A$��A��NA���A˾�@�v�@��"@��A���A���C���A��A�A�S�A�;,AT�cB�\A��{A��YA�x�AiKA�H�A��AB�@� �A�@� �A�R�A�:,AϊM@��A���A*�Aj��Ap��?��A.�A�x�@��MAZ�TB�B	EDA�|�>��,A��bA�fB?hAπ}A��3A��aAX��AÇ�@�)Aʌ�A�u@�A�7@��@@a��A\��A��@��t@EBC���A��7A��lAۇ�A��WBMAZ<�@�/XA�bzA՗YA�A#|A��BA��Â�@���@̹�A.A���A��C��<A���A UA��A�x�AU4ZBA A���      +   
   (                  *   "                  	            1      #   -      G   /            *            !   !   T            	         G                                 
      *                  #      	               )   	      !            3                                                   #               '   +            /      !            =                     )         #                  !                              3                     !                                                                                       #               /                  9                              #                  !                              )                              O�O��~OUCFO{zhO �rNeU�N>��NI�N>�OV��OO��NCM�N~��OoK�O t|Na�jN�TN�RN�U�NO��OM�N�Of#�OP�OtO�O߾@O{f�OT�sNJ�N/X�P:�\O~ӆN�N��?O^��OX�P�ԝO="N�.O97�NÖ�Nc�N��O�ӸO7�nN<��O�8�Nn�$N߽M��(N��BN�O��{N�cN��0M�'Og��Oy-N�0�O9^AO��Oo�O�E2N���N��gN�N�EO�Q�OF��O��Nx�*Ow��O
v�  n  �  �  v  e  O  �  X  �  �  (  �  ;  c  e  L  �  p  �  ^    �  �  0    	  �  6  �  E  �  �      �  �     B  �  l  �  �  )  i  �  �  �  d  �  '  �      �  �  �  �  m  �  �  p  o  �  �  +  >  �  H  �  �  8  �  �<T��;��
<o�e`B;�o;D��:�o%   ��o�#�
��`B���
�o�D���u�T���u�u��o���
�C���t���`B�t���j��h�����ͼě��������ͼ����o��/����P��P�+���C����+�#�
�<j�C��C��t��\)����P��P����w�#�
�#�
�,1�P�`�0 Ž@��@��@��@��H�9�T���T����7L��o�����t���^5������P��������������������������
#/<KQRLH?/#
�����������������������_gt|������������tg`_������������������������������������������������������������_afmnrtmaa[[________8<FIUW_ZUIB<88888888���
!!!
���������������� ��������knnz����znkkkkkkkkkk������������������������������������������������������������36BDDHCB65*/33333333?BN[]dgjgf[NJB>9????7BO[chmlh_[OMCB=7777���

��������#*/5<<<:/#��������������������&*/6CFLOOPOMC6*%&&]afmz��������zma[XY]��������������������mrz����������zmicdgm9=HUan������naUH@;:9#)-6BO[hqqnb[OHB5)&#QTUamyz������zmaYTRQ����������������������������������������6<Haz�������zmaTD:56gt�������������tg^]g��������������������dhnst~���������theddEO[ht�����|mh[OIHIEE%),5BNQW[^[YNB3)&#$%0=Nck���������t[N4/0��������������������������������������������������������������������������������������������������


���������������������������������������������������������������������al{������������{nb_a��������������������//<=?@</-)//////////$fgjot�������tkggffffX[gtuwtg[QXXXXXXXXXX#<UbegbXPB<0������������������������������������������������������������5<?IUbinqsnkbUIB<625qt������������tmjkq��������������������������
������������)6AA7)���������������������������������������yz�����������|zyyyy���!$�����
#,/35/-&#"
����������������������!&),+)" ���������������������stty������������{vtsyz���������}zzyyyyyy����������������������������������������Z�P�N�A�?�N�Z�g�s����������������s�g�Z����ųūŪũŮŹ�����������������������ƿa�T�R�J�G�F�G�I�Z�`�m�y�������������y�a�A�7�*������(�5�A�G�N�W�Z�\�_�W�N�A�����������
���#�/�4�1�/�'�#���
������ڼּμּڼ�����������������}������������������������������t�p�t�āčĚĤĚĚčā�t�t�t�t�t�t�t�t�ʼü��������ʼּڼּܼѼʼʼʼʼʼʼʼ��Z�N�G�A�7�6�A�N�Z�g�������������y�s�g�ZĿĳĦĚčĄāĊčĚĦĳĿ������������Ŀ���������������������������������������޻�ݻܻһٻܻ�������������������������������������������	��	���������ɽݽԽнĽ����ĽĽнݽ�������������ݿT�N�J�T�`�m�y�{�z�y�m�`�T�T�T�T�T�T�T�T�y�q�p�n�y�����������������������y�y�y�y�e�Y�Z�_�e�l�r�~�����������~�}�r�e�e�e�e��߽ݽ׽ݽ�������� ����������H�?�<�7�<�G�H�J�U�a�\�U�H�H�H�H�H�H�H�H�@�6�4�/�,�2�4�@�M�Y�f�q�{�~�v�r�f�Y�M�@�����������	����"�&�)�&�"��	������ƸƳƮƬƳ���������������� ����������������������������$�)�0�6�5�0�*�$��������� �
��#�$�0�<�D�C�F�@�<�0�#���ù����������������ùܹ����
����ܹϹ����v�s�q�s�y�|�����������������������������������������������	���� ��	������ƎƆƁ�}ƁƎƚƤƣƚƎƎƎƎƎƎƎƎƎƎ���������������������������������������������ݿͿʿѿݿ����5�Q�g�p�l�N�5�����������������������&�$��������	������ؾ޾��������	��"�"���
�	�<�6�/�#�!�����#�/�<�>�H�U�T�J�H�<�<�x�l�n�q�o�p�x�������������������������xÇ��z�t�n�q�zÇÒì÷ùûûùðìàÓÇ�
����²¦²������H�R�K�V�T�/�#�
�r�l�e�c�e�e�n�r�~�����������������~�r�rŹŸŭŠśŔőŔŠŭŹ��������������ŹŹ�x�t�l�_�Z�S�F�E�F�S�l�x��������������x����������!�-�0�:�-�!�����������������	��	��"�%�$�"�����������g�c�[�U�V�[�g�t�t�g�g�g�g�A�:�<�A�S�_�l�x�������������������x�S�A�ֺԺɺĺɺͺ����������������FFFFFFF$F1F4F4F1F$FFFFFFFF�(������ڿݿ������,�5�D�N�O�B�5�(�;�:�0�;�A�H�T�U�a�k�a�T�H�@�;�;�;�;�;�;�[�Z�T�[�h�t�|�v�t�h�[�[�[�[�[�[�[�[�[�[������������������������������������������ƹƳƧƞƞƧƳ�����������������������������������	����	���������������������лû����»ƻлܻ�����������ܻ��)�"�"�$�)�*�6�B�H�M�O�V�O�F�B�6�)�)�)�)����
������'�)�6�8�6�)�(�������ּҼԼּ�������������㽞�������������������Žн��ݽн������������������*�C�O�Y�U�W�O�C�6�������������ż�����������������������àßÖÏÈÁÇÉÓàéìõý����üùìà��f�Y�R�L�K�N�W�c�r�������������������'�������'�-�4�@�M�Y�\�Y�T�M�@�4�'��������y������ʼ�������	����ּ�����������������������������ŔŏŇŇŇňŎŔŠŭŮŭŬŠşŔŔŔŔŔE7E/E*E*E*E7E=ECEPEYE\EiEkEiE\E[EPECE>E7�H�B�A�H�L�T�^�a�m�u�x�m�l�a�V�T�H�H�H�H�l�i�P�G�L�S�`�l�����������������������l�����������������
���"�!���
������������������ĽĿ���������������
���������׾Ӿʾ��ʾ׾�����������׾׾׾׾׾��S�I�B�8�0�$��$�0�I�V�b�o�}ǁǄǁ�{�o�S�a�f�h�a�\�T�H�F�;�3�/�*�'�/�;�H�T�a�a�a Z . 7 E R ! J b J < F ] 2 A 6 I 4 @ /    " ) ) " ) E 2 2 B V F D R @   K  Q e A G / ) ) m c S @ Z h c / 3 H Q "  j U C H Y W X < ` ; > 4 Y W ,    q  r  �  �  /  k  u  �  h  �  �  �  y  �  %  p      y  Y  �    �  ?  �       �  g  P  �  &  	    �  �  ~  +    �  �  0  �     �  �  k  �  .  "  �  P  �    �  ,  �  �    �  u  D  �  �  �    �    �  H  �  ,  +  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  Bl  n  k  h  f  c  _  W  H  6  '      �  �  �  �  y  b  G  )  �  �  �  �  �  �  �  �  o  Q  (  �  �  o    �  >  �  �  �  �  �  �  �  �  �  �  �  �  t  a  R  I  A  9  1  *  #  '  +  �  �  �  �  �  �  �  (  U  p  u  i  K    �  �     �     �  e  V  H  :  9  B  2      �  �  �  �  �  h  G  J  v  j  I  N  O  N  K  E  <  ,      �  �  �    ]  :    �  �  �  �  �  �  �  �  �  �    u  l  d  \  T  \  n  �  �  �  z  j  [  X  O  G  ?  7  /  &             �  �  �  �  �  �  �  �  �  �  �  �  ~  n  ]  G  /    �  �  �  �  �  k  Q  5     �  �  �  �  �  �  �  �  �  �  q  :  �  �  l    �  (  v  �  W  �    '      �  �  �  �  �  �  ^  .  �  �  [  �        �  �  �  �  �    u  k  `  T  H  <  K  l  �  �  �  �  �  �  ;  1  &      �  �  �  �  �  �  �  q  U  6    �  �  �  �  c  O  <  )      �  �  �  �  �  �  �  �  �  �  �  �  �  �  R  Z  a  d  d  ]  P  ?  (    �  �  �  �  x  Z  :      �  L  L  K  J  D  =  6  +        �  �  �  �  �  �  �  �  b  �  �  �  �  �  �  �  �  �  p  Z  G  4  (    �  �  r  [  C  p  n  i  c  Z  O  ?  0        �  �  �  �  W    �  {    �  �  �  �  {  e  P  =  +      �  �  �  �  �  �  �  �  �  �  �    U  a  d  b  ^  W  O  H  A  :  3  +  !          U  �  �  �  �  �          �  �  o  #  �  m  �  $    L  �  �  �  �  �  �  �  �  �  z  h  S  ;    �  �  �  I   �   �  o  |  ~  �  �  �  r  [  <    �  �  `    �  K  �  t  �  a  �  �  �      (  /  .  *      �  �  I  �  �  E  �  r  �        
     �  �  �  �  �  �  Z  '    �  �  z  9  �  �  �  	  	  	  	  �  �  �  y  D  �  �  e    �    j  �  0  `  5  N  Q  �  �  �  �  �  �  �  �  K    �  v  !  �  $  T  `    6  /  '      �  �  �  �  �  �  g  =  
  �  �  /  �    �  �  �  �  �  �  �  �  �  �  �  �    7  W  }  �  �  �     �    3  G  M  T  U  Q  N  H  B  :  *      �  �  �  �  �  �  �  �  �  �  �  c  ,  �  �  �    �  �  o    �  D  �  �  �  �  �  �  �  �  z  h  U  @  *    �  �  �  �  }  m  ]  M  �  �  �  �  �  �  �    	  �  �  �  �  �  �  g  K  8  3  '      	  �  �  �  �  �  p  �  �  �  h  H  (    �  �  �  �  �  �  �  �  �  �  }  x  p  \  G  -    �  �  �    Q  S  6  �  �  �  �  �  �  �  ~  \  4    �  �  q  0  �  i  �  :  1  �          �  �  �  h  ;  !  �  �  l    �    r  �  u     *  5  >  B  ?  8  .    	  �  �  �  Q    �  �  f  2    �  �  {  q  g  Y  L  >  /      �  �  �  �  �  �  q  \  H  T  ^  g  k  i  a  W  K  >  1  '       �  �  �  H    r  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  b  ?    �  ^   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    '  )  '         �  �  �  S  �  �  �  #  S  w   �  �  -  U  h  e  [  M  ;  "  �  �  �  W    �  d  �  �  �  �  �  �  �  {  o  b  T  D  3  !  
  �  �  �  �  |  W  .   �   �  �  |  l  ]  T  K  C  8  *    
  �  �  �  �  f  =    �  �  �  �  �  �  �  �  �  �  �  p  \  ;    �  �  }  =    �    d  Z  P  F  <  3  )           �  �  �  �  �  �  �  �  �  P  [  f  s  �  �  �  �  �  �  �  t  M  &  
  �  �  �  �  �  '        �  �  �  �  �  �  �  �  }  e  M  5       �   �  �  x  e  S  @  *    �  �  �  �  �  w  Y  ;        �   �   �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �          �  �  �  �  �  n  V  8    �  �  �  d  5    �  �  �  �  �  �  �  �  z  h  R  8    �  �  �  �  �  �  �  q  �  �  �  �  |  f  O  6       �  �  �  �  �  n  A    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  D    �  �  y  Q  +  �  �  +  l  �  m  `  O  >  +      �  �  �  �  {  ]  B  *    �  �  d  &  �  y  \  i  j  >    �  �  �  f  C  6  6    �  �  �  O    �  �  �  ~  n  X  ;    �  �  �  J  �  �  M  �  �    �    p  Y  C  -    �  �  �  �  �  �  p  L  @  )  �  �  8  �  *  o  b  U  H  :  ,      �  �  �  �  �  �  �  v  a  H  .    k  }  l  E  $  &  �  �  �  U    �  �  L  	  �  l    �   �  �  �  �  �  �  �  �  �  �  �  �  q  A    �  �  [    �  �  *  *  +  #      �  �  �  �  �  �  s  [  C  (    �  �  �  $  2  =  ;  1  #    �  �  �  y  I    �  �  �  t  T  =  H  �  �  �  �  �  t  h  W  D  1      �  �  �  �  �  �  �  s  =  G  =  +    �  �  �  k  9    �  �  ^  %  �  _  �  �  �  �  ~  s  e  Q  :  %    �  �  �  m  1  �  �  [    �  Y  �  N  s  �  �  �  �  �  �  �  �  �  �  �  �  B  �  �  4  �  �  8  %    �  �  �  �  �  y  \  ?  "    �  �  �  �  �  _  8  �  �  �  �  �  �  �  q  9  �  �  W  	  �  t  !  �  `   �   b  �  �  �  i  I  (    �  �  w  D    �  �  )  �  �      "