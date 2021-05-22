CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��t�j~�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��/   max       <49X       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?nz�G�   max       @Fo\(�     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @vx�����     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @Q            �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�            7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       ;�o       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��p   max       B2��       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B2!       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C��h       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�]=   max       C���       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          `       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P�)�       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��a@N�   max       ?ԧ��-       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��;d   max       <49X       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?u\(�   max       @Fo\(�     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vx�����     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @Q            �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�7@           Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Dj   max         Dj       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��)^�	   max       ?ԛ��S��     �  ]            F      5   `               
      <         	   !   5   
      =      .      3      #   3         	      
   +   %            
   >         
                              $   	   	                  J            	   
      .   N�N��O�r�P/�N࠘P�+�P��N*w�O[��OQ'�Nt�O.*�N6E�P���O��Nf�VOW�Ol@�O�[�Nn�N�1P_�,O�z�PH�OƏO�O���P[��O�O�mM��N�j�ONh�N��P&�Oò�O��RO2��N5lgN�EdP��eN�jrN�uN/�vN�/9O:� O�OY�NY4�N\�Nԉ�OC,O�5O|�
O/�O� N�{�OV��N��9O2� Ow��O��N��zOImqOP"N�U�Np��N)��O�?tN�A�<49X;��
;�o:�o�o�o��o���
�ě���`B�t��u���㼣�
��1��1��1��9X��9X��9X��j��`B��h�����o�o�+�C��\)�t���P���#�
�#�
�,1�,1�,1�,1�0 Ž8Q�@��@��@��D���H�9�H�9�L�ͽT���]/�]/�e`B�ixսixսixսy�#�y�#��7L��7L��t����㽝�-���w����{��{�����/��/��������������������
 )*-,*)	



)5BOhmoh[OB65960)
)8gt�~m[N5)/;<BNX[glggtvtg[NB5/%4+an���������aH/%��#0n{����{n<0
��������	�������������*16CN\]OC6*
#/<HNQND</,(#
����������������������������������������ehux����ukh\eeeeeeeew�����������wpvw5=BO[_hiusrh[OB@;865W[[ht{{wthc[WWWWWWWW��������������������GHMTamz�����zwmaTMGG}������������|{{|{}}#*/##!��������������������amz�������������l`^a')67BOYY]cg[OB52,,('�0IUbgjjURPE<0���������������������SVajnz��������zmaYUSkt�������������wkkgk�B[hu���p]B60)���#/HXinxznUH/#	&+5BNVWUVIKB5)"��"&#����������������������������� ���������� )07BN[b�������g[5 RZmz�������zumaTPNOR��������������������������  ������������")6;;:6))&""""""""""[[[htu�������th[[W[[gy��������������|gbg�����

�������������������������)67;6)"��������������������nz��������������zvnn����������������������������������������_ht��������th_______LNPY[gjppg[NLLLLLLLL�����������������������*5CDB)
�����������������������������������{��������������{yx{�����������������������������������������	

 �������fnu{�������{oncffff#(-/4<BHKLKHB</(#"!#��&'$ ��������/3289=>D</,$
 �������������������aanz�����������znd`a������������������������������������������������������������fgnt����tgffffffffff5BN[gqng_UNB5)������������tx�������������������������������������������ɺ��ƺɺպֺ�����������ֺɺɺɺɻ��!��� �F�S�X�f�q�s�_�S�F�A�:�/�#��5�(�������(�N�c�p���������s�g�N�5�������������������������������������¿��
�����������
�#�<�H�`�b�v�i��y�m�i�H�
�����s�[�P�I�=�C�Q�s������������������t�s�i�g�b�g�t�~�{�t�t�t�t�t�t�t�t�:�.�)����������	�"�.�<�@�@�>�>�?�=�:�[�V�B�6�-�,�-�0�6�B�O�h�s�v�t�h�]�g�`�[�U�T�N�H�H�H�T�U�V�a�b�b�a�Y�U�U�U�U�U�U�x�s�w�j�j�l�x�������������������������x�������������������������������������������������;�H�a�m�������������m�T�H�	��m�`�T�V�]�`�j�m�y���������������������m���������������������������������������ؽнĽĽ����������Ľнݽ�������ݽ�ā�~�t�l�a�\�`�h�tāčĚģĦĭīģĚčāĳħġğĦĿ�����
��&�%�����������Ŀĳìêêìù������ÿùìììììììììì�������������ʼμּڼּʼ���������������ƽơƖƒƊƑƚƳ��������!��"�������ƽ�������������ʾ׾����	������ʾ����������{�z�}������������ݽνý��Ľ������ɺ��������������ɺܺ�����������ֺ���������������������"�/�:�?�?�<�/�"���¦¿��������������¿²¦�i�g�v���y�^�Z�K�Z�������������������i�Z�M�E�>�;�<�A�M�U�Z�f�i�y�}�������s�f�Z��������������	�"�;�T�a�v���}�a�V�/�"��'��'�2�3�@�A�E�@�3�'�'�'�'�'�'�'�'�'�'�����������������������������������������T�F�F�H�N�T�a�m�y�z����������}�z�m�a�T��ؾ������	��	��������������׾����������ʾ׾��	� �(�(� ������5�(�����(�5�A�N�g�����������s�N�A�5ŲŠŔŒŒŗŠŭ����������������������Ų����������������û̻лܻ�лû������������������������������������������������Ҽ@�@�@�D�L�M�U�X�Y�f�f�f�d�e�c�Z�Y�M�@�@�/�
����� ��0�nŇŘŉňŔťūŇ�`�I�/����������������������������������������àß×ÓÎËÌÑÓÜàâèëìííìààìâáéìùÿÿùõìììììììììì�Y�N�O�R�Y�e�k�r�v�~�����~�t�r�i�e�c�Y�Y�������������ĿϿѿݿ�����ݿѿǿĿ����������������������Ŀѿݿ�����ݿѿĿ����� �����)�6�B�O�U�O�K�D�B�6�)��������'�-�3�6�3�/�.�(�'������ĿĺĳĦĦĥĦĳĿ������ĿĿĿĿĿĿĿĿ�лʻû��������ûлֻܻ������ݻܻл��t�i�t�|ćĒĚĢĦĳĴĻĻķĳĦĚčā�t���������������'�4�4�4�3�'���	��Y�S�M�@�9�<�F�M�Y�f�r��������������r�Y�&���������'�4�@�M�Y�f�e�M�@�4�&�����������������������ʼּݼؼռʼ�������߼��������������������Ŀ������������Ŀѿݿ�����������ݿĻ����������������ûлӻлϻû»���������E�E�E�E�E�E�E�E�E�E�E�FFF F#FFFE�E𽅽y�������������Ľн����ݽнĽ�����ED�D�D�D�EEECEPE\E�E�E�EuEiESECE*EE�Z�Z�N�A�N�R�Z�g�s���������s�n�g�Z�Z�Z�Z�ù¹����������ùϹܹ����
�����ܹϹú3�(����'�3�@�Y�b�r�����������t�Y�C�3����
���$�*�,�0�2�0�$������������$�+�0�3�2�3�0�$�������������������������������n�Z�R�O�P�T�U�a�n�zÇÔåëèàÓÇ�z�n�<�<�5�<�>�H�U�a�m�j�a�a�U�H�<�<�<�<�<�< Z V L 7 p H 8 [ Y 8 ~ H R J % 9 F  I K [ B l g 9 4 & _ S a ] U > . / [ N H ? � ) T � \ 5 k k c � E . � K - i F ]  5 [ L l J > | F F W O 2  O  �    (  X  2    \  �  �  �  �  f  �  �  �  f  �  2  <  1  �  �  �  a  5  9  =  �  r    �  �  �  �  �  j  �  O    ;  �  <  r  �  �  l  d  �  �  �  }  L  �  N  [  %  �  �  �    ^  �  �  �  �  �  h  �  �;�o�D���D����C��o�aG������D����C���1��o���ͼ�9X���
�'�/��h�aG����P����/��9X�'���P�`����Y���7L��1�}��w�8Q�y�#�L�ͽ��罟�w��t���%�]/�T������q���q���e`B�e`B��o�ixս�\)�}�y�#��\)���
��\)��j�����O߽�hs��-��\)�\��������� Ž������vɽ����;d���
=qB/�B��B��BA�BJ�B�/B&U�B��B0	kB.B JB Q�B2��B�B�B�B!<lA��pB4B�{B TBp�BE�B&IB#MA��!B
�uB6�B}$Br�Bb�B��B;�B{�B	@�A��$B��B��B��BG�B#=BB�DB%�B!r�BZ	B*6�B��BI�B�6B�-B�%B�Bh9B)j�B+WfB-�!B��B(�B�BB��B;�B��Bk�B�kB��B|�B	�9B�xB
�B=LB��B�B@dB4bB��B%��B�eB0:�B�B �B CJB2!B��B�GBÊB!+.A��B#�B�!B��B J�B[�B&� B#<�B 5�BI�B��BAB=B?B�7B?�B?�B	�'A��IB�TB��BGlB��B��B@�BB�B��B!>�B�B)��B�BAxB	2�BC�BϳB�BE�B)[qB+@XB-�bB@B)"�B5GB6sB?�B�	B��BA�B�B>�B	��B~B
�&A�c=@CP@�'�A�ǱAs��A{A�� A�[dA_�EA�p3A�Y@�m"AK	A��	An�gA�7�A)l*A��vA���A�A�@�/8BAP� A%/|@5��A�I�A���A�f�A@;AA�l�?�{A��MA�2�AW��AW}�A���A���@���A��@�f�A�v�A���A�sA��?��A{(�Ax��Aւ�?� �A��@��A�ě@�i�@��?@ЅA@��A��A|S4@��C��hA%�C���A��>���?�<�B	�B	��A�P�A�O�A�@8A���@K��@�2UA�|�As?zA�A�t{A�wA]��A��A��@��:AK�A�~�Ao)�A�uA*��A�qrA��TA�|�@���BI�AO�CA!�@4<5A���A��A�z�A?�A�?�x�A�}�A��AW(�AY�A���A��@���Aϋx@֋�A��A�z�Aʉ�ÂJ?���A{CA{��A��?�>�Aᪿ@�ݳA��@���@��(@�D8@�\�A"Az��@��GC���A&ښC��A�~�>�]=@��B	�@B	��A�}AȇNA��            F      6   `                     =         	   "   5   
      =      /      4      #   3         	         ,   %            
   >         
   	      	                     $   	   	                  K   	         	   
   	   .            !   -      ?   ;                     ;               !         1   !   -      %      C   !   %               )   #   !            9                                                               '         !                                 ?   9                     1                        1   !   '            =      %                                 -                                                                        !               N�N���O>�O�g7N�s�P�+�P�)�N*w�O[��OQ'�Nt�O.*�N6E�P\f�O��Nf�VOW�O+�~O�4�Nn�N�1P_�,O�z�O�kNTP�O��OJ=PQ�O��O�mM��N�j�O@]yN��O�o�O�l&O��!N��N5lgN�EdPO�N�jrN�uN/�vN�/9O:� O�OY�NY4�N\�Nԉ�N�:�O�5O]X�N�XO� N�{�OV��N��9O2� OGw\O�N��zO;��Op�N�U�Np��N)��O�ԃN�A�  B  3  j  &  �  g  �  W  }  �  �  �  �  �  /  B  ~     Q  �  �  	�    ]  �    �  X  �  I  �  s        `  �  �  i  u    +  �  u  O  �  �  �  �  �  �  y  M  s    �    �  �  �  �  �  4  e  �  4  B  �  
q  �<49X;D��%   �����D���o�D�����
�ě���`B�t��u�������1��1��1��`B�\)��9X��j��`B��h�C��#�
��w���C��0 Ž\)�t���P��w�#�
�m�h�@��0 ŽT���,1�0 ŽaG��@��@��@��D���H�9�H�9�L�ͽT���]/�]/��%�ixսu�m�h�y�#�y�#��7L��7L��t�������"ѽ��w���罩�置{��{�����;d��/��������������������'),*)(36BOY[efhihg[ODB;?:3)5BNZ__ZNB5)=BENV[dd[NB?========%4+an���������aH/%���#0bn{����n<0
��������	�������������*16CN\]OC6*
#/<HNQND</,(#
����������������������������������������ehux����ukh\eeeeeeeey������ �����vuy5=BO[_hiusrh[OB@;865W[[ht{{wthc[WWWWWWWW��������������������QT^amzz~}zymmaTPJJQ��������������������#*/##!��������������������amz�������������l`^a')67BOYY]cg[OB52,,('
0IUbfihbURG<0#
�������� �������������afmz�������zmd\XWXarty��������������tsr�B[h��vtjOB61* ��#/<HMU`b\P</#&+5BNVWUVIKB5)"��"&#����������������������������� ����������@BIN[gt�������tg[OF@QUamz�����~zomaVSQPQ����������������������������������������")6;;:6))&""""""""""[[[htu�������th[[W[[����������������lhl������

�������������������������)67;6)"��������������������nz��������������zvnn����������������������������������������_ht��������th_______LNPY[gjppg[NLLLLLLLL���������������������#)3/)�����������������������������������y{|������������|{zyy�����������������������������������������	

 �������fnu{�������{oncffff#(-/4<BHKLKHB</(#"!#�
$$!���������
#'+,-,#
������������������`abnz�����������zne`������������������������������������������������������������fgnt����tgffffffffff5BN[gjmg_UNB5)������������tx�������������������������������������������ɺúȺɺֺٺ������������ֺɺɺɺɻ-�+�!��!�"�-�:�B�F�N�S�`�j�_�]�S�F�:�-�A�5�*�����(�5�A�N�\�e�k�q�n�g�[�N�A�����������������������������������������
�����������
�#�<�H�`�b�v�i��y�m�i�H�
�������s�T�N�G�I�Q�c�s�������������������t�s�i�g�b�g�t�~�{�t�t�t�t�t�t�t�t�:�.�)����������	�"�.�<�@�@�>�>�?�=�:�[�V�B�6�-�,�-�0�6�B�O�h�s�v�t�h�]�g�`�[�U�T�N�H�H�H�T�U�V�a�b�b�a�Y�U�U�U�U�U�U�x�s�w�j�j�l�x�������������������������x�������������������������������������������������!�/�H�T�a�m�z������a�T�H�;��m�`�T�V�]�`�j�m�y���������������������m���������������������������������������ؽнĽĽ����������Ľнݽ�������ݽ��t�p�h�e�c�g�h�tāčĘĚĦĦĦğĚčā�t��ĿĳĮĪĩĳĿ������������
��������ìêêìù������ÿùìììììììììì�������������ʼμּڼּʼ���������������ƽơƖƒƊƑƚƳ��������!��"�������ƽ�������������ʾ׾����	������ʾ����������~�|��������ݽ�����ݽʽ������������ɺ����������ɺκֺ�ߺֺɺɺɺɺɺɺɺ�����������������"�/�6�<�<�7�/�"��	����¦¢¦²¶¿��������������¿¦�k�h�w���z�f�R�Z��������������������k�M�J�A�>�>�?�A�E�M�Z�f�s�}�������s�f�Z�M��������������	�"�;�T�a�v���}�a�V�/�"��'��'�2�3�@�A�E�@�3�'�'�'�'�'�'�'�'�'�'�����������������������������������������T�Q�H�F�G�H�P�T�a�m�z��������|�z�m�a�T��ؾ������	��	��������������ʾȾɾӾ־ܾ����	������	�����A�5�%����(�5�A�N�g�v���������s�Z�N�AŵŠœœřŠŭ������������������������ŵ�������������������»����������������������������������������������������������Ҽ@�@�@�D�L�M�U�X�Y�f�f�f�d�e�c�Z�Y�M�@�@�����	��0�U�nňŋ�y�xņł�n�U�I�0�����������������������������������������àß×ÓÎËÌÑÓÜàâèëìííìààìâáéìùÿÿùõìììììììììì�Y�N�O�R�Y�e�k�r�v�~�����~�t�r�i�e�c�Y�Y�������������ĿϿѿݿ�����ݿѿǿĿ����������������������Ŀѿݿ�����ݿѿĿ����� �����)�6�B�O�U�O�K�D�B�6�)��������'�-�3�6�3�/�.�(�'������ĿĺĳĦĦĥĦĳĿ������ĿĿĿĿĿĿĿĿ�лʻû��������ûлֻܻ������ݻܻл�čĂċčĖĚğĦĳĵĶĴĳħĦĚčččč���������������'�4�4�4�3�'���	��Y�W�M�@�=�@�J�M�Y�f�r������������r�f�Y�4�+�'����!�'�4�@�M�Y�b�c�Y�M�G�@�4�4�����������������������ʼּݼؼռʼ�������߼��������������������Ŀ������������Ŀѿݿ�����������ݿĻ����������������ûлӻлϻû»���������E�E�E�E�E�E�E�E�E�E�E�FFF F#FFFE�E𽎽������������Ľн۽ݽ��ؽнĽ�������E*EEEEEEEE*E7ECEPEWETEPEHECE7E+E*�Z�Z�N�A�N�R�Z�g�s���������s�n�g�Z�Z�Z�Z�Ϲǹù����������ùϹܹ������
����ܹϺ3�+��'�3�@�Y�c�r���������������r�Y�?�3����
���$�*�,�0�2�0�$������������$�+�0�3�2�3�0�$�������������������������������z�n�Z�R�O�Q�T�U�a�n�zÇÓäëèàÓÇ�z�<�<�5�<�>�H�U�a�m�j�a�a�U�H�<�<�<�<�<�< Z O ' + 2 H 6 [ Y 8 ~ H R K % 9 F  A K [ B l b 5 & * f A a ] U 6 . 5 Y J N ? � 6 T � \ 5 k k c � E . d K / R F ]  5 [ = B J 9 x F F W M 2  O  �  N  P  �  2  �  \  �  �  �  �  f  
  �  �  f  l  ]  <  1  �  �  P  q  �  �  �  2  r    �  �  �  .  �  H  �  O    a  �  <  r  �  �  l  d  �  �  �    L  �  �  [  %  �  �  �  �  S  �  �  �  �  �  h  �  �  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  Dj  B  V  i  y    �  x  ]  A    �  �  �  �  _  :    �  �  �    &  1  +    	  �  �  �  d  ,  �  �  �  e  =    �    :  9  4  6  Q  h  j  h  `  U  D  .    �  �  �  �  �  �  �  v  �  M  �  �  �       $  "    �  �  k    �  8  �    �  �  f  �  �  �  �  �  �  �  �  �  �  {  o  b  S  D  /  �  �  �  g    �  �  �  �      �  �  �  �  t  W    �  �  �  �  �  �  �  �  �  Y  %  �  �  y  P  1    �  �  d  �  b  �  }  3  W  U  S  M  A  5  '    
  �  �  �  �  �  �  j  P  2    �  }  w  q  k  d  [  K  9  '      �  �  �  �  �  d  6     �  �  �  �  �  �  �  �  �  �  �  �  W  "  �  �  �  _  1    �  �  �  �  �  �  g  E  !  �  �  �  �  h  B    �  �  �  �  f  �  s  c  b  ^  G  .    �  �  �  �  �  �  �  �  �  �  �  �  �  ~  z  v  r  n  k  `  S  F  8  +       �   �   �   �   �   �    B  w  �  �  t  _  C    �  �  _  �  �  ,  �  I  �  �   �  /  )    	  �  �  �  �  b  2  �  �  �  j  H  #  �  �  �  �  B  9  /  %        �  �  �  �  �  �  �  ^  ;     �   �   �  ~  y  t  m  f  ^  S  I  :  +    �  �  �  s  M  *  
   �   �  �  �            �  �  �  u  >    �  C  �    r  �  �  �  �    2  J  Q  J  9    �  �  ]    �  �  9  �  �  (  H  �  �  �  �  �  �  �  }  c  I  /    �  �  �  �    ^  <    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	�  	�  	�  	r  	'  �  �  8  �  m    �  �  Q  �  L  �  �  �  �      �  �  �  �  �  �  �  k  O  .  �  �  �  �  �  g     �  I  H  Z  M  3    �  �  �  q  1  �  �  �  �  G  �  H  _   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  \    �  �  _    �            �  �  �  k  (  �  �    �  &  �  �  �    �  �  �  �  �  �  �  �  �  k  6  �  �  U  �  �  W  ,    �  T  S  ;      �  �  w  L    �  �  y  :  �  �  3  �  f   �  S  �  �  �  �  �  }  F    �  a  
  �  D  �  x  �    /  D  I  D  <  4  *        �  �  �  �  W    �  �  >  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  c  R  ?  *    �  �  �  �  �  �  �  v  f  U  C  )    �        �  �  �  �  �  k  S  D  1    �  �  |  <  �  �  �    �  �  �  �  �  �  �  y  f  R  >  (    �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  S    �  C  �  g  ?  X  _  `  Y  L  2    �  �  i  &  �  �  .  �  Z  �  T    �  �  �  �  �  �  Y  0    �  �  Z    �  �  b    {  �  '  J  `  m  p  o  k  e  [  y  �    v  m  X  *  �  �  K  �    i  ^  K  .  .  5  ?  @  <  9  5  0  $      �  �  �  �  �  u  o  i  P  5       �  �  �  �  w  [  ?  $    %  +  �  r  �  �  �          �  �  �  �  z  ?    �  ~  �  u  �  }  +    �  �  �  �  �  n  Q  R  :  	  �  �  \    �  �  Z    �  �  �  �  �  �  ~  b  F  *    �  �  �  �    e  ]  U  L  u  t  s  o  j  d  ^  V  N  E  <  .      �  �  �  k  3  �  O  @  1       �  �  �  �  �  ~  b  D  "     �  �  �  	  >  �  �  �  �  �  d  A    �  �  �  �  _  7    �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  b  F  *    �  �  �  e  ,   �  �  �  k  L  1  	  �  �  v  ?  
  �  �  �  D  �  �  L  �  B  �        �  �  �  �  �    '  <  B  =  7  2  +  %      �  �  �  �  |  t  k  a  X  M  B  7  (    �  �  m  $  �  �  �  �  �  �  �  j  C    �  �  �  T    �  �  `  C  *      =  .  G  _  U  X  l  N  )  �  �  v  )  �  �  "  �  M  �  D  M  ;  (    �  �  �  �  �  ~  f  K  +    �  �  b  P  C  9  o  r  r  m  c  V  G  3    �  �  �  P    �  k    �  >  +  �  �  �  �  �  �  �  �  �  �  �  �  v  [  @  &    �  �  �  �  �  �  x  m  `  P  A  /      �  �  �  �  v  [  @  #      �  �  �  �  �  �  d  D  $    �  �  �  e  9    �  �  |  �  �  �  r  `  J  ,  	  �  �  �  [  -    �  �  �  ~  l  [  �  �  �  |  y  u  r  k  b  Y  P  G  >  :  @  F  K  Q  W  ]  �  �  t  Y  @  $  �  �  V  
  �  ]    �  d  �  �  ;  �  {  �  �  �  �  �  �  �  �  �  n  G    �  �  �  o  2  �  �   �  �  �  �  �  �  �  �  v  �  c  "  �  ^  
�  
  	x  �    3  D  4  0  ,  %      �  �  �  �  �  �  �  �  �  f  E     �  �  `  c  W  H  7  "  	  �  �  �  b  '  �  �  ~  A    �  �  =  �  �  �  �  j  4  �  �  �  O    �  |  4  �    �    �  �  4  ,  %    	  �  �  �  �  �  �  �  h  M  2    �  �  �  �  B  N  Z  ]  _  Y  Q  C  2  #      �  �  �  �  �  �  �  n  �  �  �  �  �  �  �  y  b  K  (  �  �  �  a  +  �  �  ~  B  
U  
j  
M  
+  
   	�  	�  	X  	  �  �  -  �  `  �  W  �    O    �  �  �  {  g  M  -    �  �  9  �  |    �     �  '  �  