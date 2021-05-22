CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��t�k       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�g   max       P��5       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <�C�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��R   max       @E�
=p��     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ҏ\(��    max       @v\��
=p     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @Q�           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�o�           7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       ;�`B       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��Z   max       B-3�       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B-�_       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >ww�   max       C��;       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >Q�6   max       C���       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          V       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�e�   max       P���       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�!-w1�   max       ?��Q�       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <T��       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��R   max       @E��
=q     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @v\(�\     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @Q�           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�            Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B>   max         B>       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?k�u%F   max       ?��t�j     �  ]   0         =      :      )      2      	   A   6            	            	            !   0                                                            
         -      %               
                  	   V         	   '      	O�j6N<k�N�)O��N��P=O�.P}bN�gP��5O`�<N�:Pa^PO˥Nm�EO>��O�v�N۱�N'AvO0�tN���N�H�O���O JN<�P�{P2"�O��O0�N��pN�0�O=��O[�(N.JLOn�N���N9�}N5N�ǼN���O���Nq-OgmN��JOH�O��N�CO��+N��	O�1�O�O�OO5�O�W)OxRyO	BKN=��O5�Oל}O��N�;fO#>N��O�iO���OL�N`�O�~MOK[N�g�<�C�<T��;�`B;ě�;��
:�o�o���
�ě��#�
�D���D���D���T���T���T���u�u��o��o��o��C���C���1��1��9X��9X��j��j�ě����ͼ��ͼ�����/��`B��`B��`B�����o�o�o�+�+�\)�t��t��t���P�������,1�,1�8Q�<j�@��T���aG��aG��aG��aG��q���y�#��o��+��C����-������/<HUZVH</#
������������������������Yanz���}zngaYYYYYYYY���������������(/7;<=;/+#((((((((((��������� 	
�������<BLOY[hhca[VOB>;9:8<"'6CUaz}~|wnZUH<7/#"
#)#
 ��#0n{�����{U<0����������
��������$),6BEIMGB6)& $$$$$$�������))%�����������"!"��������55?BNTSQNGB?85555555����������������������)5BKSTOGB)��������������������������������������������
#-/5::/&#
�����������������������MN[gnrrnge[VONMMMMMMsut��������������wtsOO[hltttnnlh[YOKGEHOHINUbinponb]UIHHHHHHmz�������������zmdhmz����������������sqzTUZam���������zmaXTT����������������{~�%).6BLNMEB;6)%#"%%%%
#%/26//#
�����
!#$"
�������������������������nt�����tmnnnnnnnnnnlmz}���������zrmhegl������������������������������������������

������������������

�����������!!	������^t��������������ti\^��������������������gt�������������tg[^gZbhn{������{unjbb]ZZ��������������������-/<HXgnu|�~znaUH<1.-�����������������������������������������������������������������������������������������������
2UbinombUI0���W[\egt}�������tg[ZXWny��������������~wkn�&)-,,,..*#�����������������������Y[agqtx|wtg[YYYYYYYY)6BO[hmmje[OB;63*)MT[gt����������pg[PM������������������������������������������������������������NUZaenqxnaYUUPNNNNNN�����
#+---,(
�����ft|�������������tmhfqz�����������zyspnoq����������������������()(&������

�����������URKH<:99:;<HLSUUUUUU��źŭŠŞūŻ������������������������
��
���#�+�/�3�/�#��
�
�
�
�
�
�
�
�H�F�D�A�C�H�N�U�V�V�U�N�H�H�H�H�H�H�H�HÇ�z�q�n�tÇÑßù��������������ùêàÇ����������������������������������������������������������0�I�R�Y�[�P�@�0���m�a�m�o�y�{�������������������������y�m���g�N�5� ����A�Z�g�y��������������������������
����
���������������������������Y�P�3�4�?�Z���������������������׾A�4�2�(��%�1�M�Z�f�s���}�s�e�[�W�M�C�A�#������#�/�<�A�E�>�<�/�#�#�#�#�#�#�����r�k�z�����ʼ����+�/�*�#���ּʼ��T�.�	������$�;�G�y�������ǿ¿����y�m�T���������������������¾����������������������������������������л����������ܻ��Z�Y�N�E�D�F�F�H�N�Z�g�s���������~�u�g�Z�f�c�^�c�f�s�����������������s�f�f�f�f�/�%�+�/�;�H�L�H�?�;�/�/�/�/�/�/�/�/�/�/��t�l�]�[�V�[�\�g�t¢ìåàÛÛßàìù����������ýùìììì���������������ĿƿĿ���������������������ݿѿɿǿ˿ӿܿ�����(�-�(������깶���������¹ùϹܹ�����������ܹϹù��:�6�-�+�+�-�7�:�F�G�I�F�A�;�:�:�:�:�:�:���� ���/�<�[�h�{Ċ��n�f�R�Q�B�)�������ĳĦēĘĨĳ�����
�<�U�K�<���������ƳƧƛƏƁ�ƁƎƚƧƳ��������������������������������������� ����������l�e�_�Y�V�_�l�x���������������x�l�l�l�l������ �$�0�<�=�G�B�=�0�-�$�����ɺ������������������ɺֺۺ�����ֺɿy�m�U�H�C�B�G�^�m�y�������������������y¿´²²²¿��������¿¿¿¿¿¿¿¿¿¿�������(�2�5�A�M�N�O�S�N�J�A�5�(��N�J�A�?�A�C�N�Z�g�g�n�g�Z�P�N�N�N�N�N�N�r�h�o�r�~���������~�r�r�r�r�r�r�r�r�r�r��
����)�-�)�)���������������������������������������������������ŭţŧŪŭŭŹ��������������Źűŭŭŭŭ������������������������,�5�8�2�)���@�?�4�<�@�L�P�Y�c�^�Y�L�@�@�@�@�@�@�@�@�w�n�b�b�[�V�a�m�z���������������������w����"��%�'�4�@�M�M�P�M�G�@�4�3�'���������������������C�P�J�C�6�������ٹ��������������ùϹ���������ӹ�����������#�'�3�@�D�L�Q�L�H�@�6�3�'���!��������!�-�:�S�Z�x�������l�S�:�!�z�y�z�������������������������������z�z������ֻܻ߻������'�,�-�4�2�'��������������Ľнݽ�����������ݽнĽ����������������y�{���������Ľн�˽��������C�8�6�*�'�������*�6�O�\�b�]�\�O�CŭŠŚœœšŹ������������������������ŭ������ݽֽݽ�����(�4�A�J�J�B�(���S�M�F�B�@�:�F�R�S�W�_�l�p�u�t�r�l�_�S�S����	�� ����!�&�#�����������{�}��~�����������������������������ĿĦġĢĥııĴĿ������������������Ŀ�I�A�=�2�=�>�I�V�b�o�v�{ǅǈǈ�{�o�b�V�I���������
����$�'�'�$�������������������ĿĸĿ������������� �����������ؿ��������������ĿϿͿĿ�����������������E*EEEED�D�D�EEE*ECEPEbElEsEiE\ECE*���������������������
�������
���񾌾������������������ʾоӾ̾ʾ����������ּּѼҼּټ������ּּּּּּּֽ����������!�.�:�G�M�P�G�@�.�!���(�5�A�N�Z�]�b�\�Z�N�J�A�5�(�"�� �(�(�(D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� 0 A � + 6 6 9 T ^ J `  ? T = b E  V L 0 : - - ` e b P H ? : > j . C - M R C X v 8 Z Z � d l p d R : b Z , J 5 V Y 4 Q J # { J ; % ` * % Q    f  �  "  =  �  :  �  W  .    �  �  �  }    \  �  P  �  
  �  `  E  �    �  r  �    �  �  '  E  b  �  K  H  (  �  }  h  
    �  �    ,  �  '  8  �  �  7  "  D  v  �  �  \  �  c  �  �  @  �  �    ?  ���;�`B��o�]/��o�ixռo�8Q�T���u������1���㽇+��o��h�C���j��t��49X��P���ͽ'0 żě��aG���\)�H�9�0 Ž#�
�#�
�Y��t��o���o���C������aG���w�'�w�'�o�8Q�}�#�
��]/���P�ixս�C���t���%�e`B��t����罇+�u��O߽�C�����Q콲-������ٽ�xս��`B�wBZB�`B��A��ZB�%B�aB�1B#~B&��B��B��B-3�Bx'B�B++B��B"��B	Bu�B��B�*B-#BA�B'~Bb�BwA�B�-B��B�:B$W7B*�FB��A���B\Bo8B�MB�,B~�B
��B"0+B
��B(�{B��B��B=�B��B�@B"&B!�`B&IZB	�5B
�BhrB!��B	T'B��B
B��BP�B��B2�B�B
��B�bB��B�BB?B�LB@mBA5B�]A���BÌB��B��B?=B&��B�B�\B-�_BFUBłB*�B�MB#�B��B��B��BǷBB�BA�B'�HBA�BA�A��FBPB��B��B$K�B*�B��A�{pB_�B��B��B�B�XB
=�B"1jB<�B(�'B�B�]BB�B��BA�B-�B!�B&�8B	FrB5}BD�B!��B	E�B��B	�IB�BA�B�B?�B��B
��B�oB�B�|B�B��A���A��A��hA� �A��B	��Ap
�A���A�e�A�dA=v�A�TAnAjnAK�N@�x A�N�AD��A�DA�e�A��Au�A��>�jh@|�GA�	A�HpBjlA��@�5B
�@)�fAk�vA��A�7dA��@yA�u8A�K�A��A��c?���A��3@̯�A��e>ww�?�YC@zB�A�x�@�X�A)s�A!<�B Q�A�A�A4�!@�CA��hA�nWA�B B�A�/�Au�-C��;A���AK�A^�A
~A�BoC���A�f�A��EA��A�4�A�qB	��Ap�tA�;A�}6A���A<��A�AAjޛAK�@��8A���ADjA��A�fȂ�AtԺA�w>Q�6@w��A�eA��B��A���@�B	��@+��Ak�@A��A���A��t@ΕA��A�-A��0A�v?ɕ�A�g�@ϭA��V>�^�?�^�@XSA���@�A)rA!�4B ĩA�~�A6�m@��A�mpA���A�<By�B�jA�AwC���A�s�AL��A��A�A�{gC��   0      	   =      ;      )      3      
   A   6            	            	            "   1                                                	            
         -      %               
                  
   V         	   (      	   %         #      )      )      ?         5   1                                    +   5                                          )               !      '      #      '                     #               !                                                ;         1   '                                    +   '                                                               '                                 !                                 O}s�N<k�M�e�O6r	N��O��O�.O�U�N�gP���OH+eNg�qP*�P
RNm�EO)*�O�v�N۱�N'AvN�'�N½�N�H�O��xNJ�QN<�P�{O�|O�9EO�N��pN��N�\�O[�(N.JLOn�N���N9�}N5N�ǼN���OV,Nq-OgmN��JOH�O��NA�EO��+N��	O��2O�O�ahO +�OW�O'I�O	BKN=��O֭O���O��N��]N��	N��O��TO���OL�N`�O���OK[N�g�  m  A  n  �  �  C  �  Q  �  �  �  �  %    �  �  �  �  �  "  �  e    �  I  M  �  �  �  �    �      �  .  �  �  �  �  ]  �    /  %  �  �  �  �  ,  -  -  �  �  P  �  E  �  -  �  j  ^  I  F  D  �  :  �    �;D��<T��;ě���1;��
��1�o��o�ě��49X�T���u�ě����ͼT���e`B�u�u��o��j��t���C����
��󶼬1��9X�C����ͼ����ě������o������/��`B��`B��`B�����o��w�o�+�+�\)����w�t���P�49X���8Q�0 Ž@��P�`�<j�@��aG��ixսaG��e`B�u�q����7L��o��+��C����署-���
#/39><;5/#	��������������������_anz���{znia________��������������������(/7;<=;/+#((((((((((�������� �����������<BLOY[hhca[VOB>;9:8<1<BHUantwvtpiaUH<4-1
#)#
 ��#0n{�����{U<���������

��������')6BCHBA6*)#''''''''���������
$$�����������������55?BNTSQNGB?85555555����������������������)5BKSTOGB)������������������������������������������	
#//65/#
		��������������������MN[gnrrnge[VONMMMMMMz���������������ywwzKOO[[[hljha[WOKKKKKKHINUbinponb]UIHHHHHHmz�������������zmdhmz�����������������xz[amz��������zmaZUTV[�����������������~��%).6BLNMEB;6)%#"%%%%

##/15/.#
��
!
��������������������������nt�����tmnnnnnnnnnnlmz}���������zrmhegl������������������������������������������

������������������

�����������!!	������gqt������������trhbg��������������������gt�������������tg[^gZbhn{������{unjbb]ZZ��������������������./<HUdpz}~znaULH<2/.������������������������������������������������������������������������������������������������#IUbejlibUMIB0X[]gv�������tg\[YXst��������������}trs $'**)&��� ��������������������Y[agqtx|wtg[YYYYYYYY16BO[hkkhhb[OIB>6611NW[gt����������sg[QN������������������������������������������������������������NUZaenqxnaYUUPNNNNNN�
#(**+*&
��������ft|�������������tmhfqz�����������zyspnoq�����������������������(((%������

�����������URKH<:99:;<HLSUUUUUUŭūŦūŴŹ������������
���������Źŭ�
��
���#�+�/�3�/�#��
�
�
�
�
�
�
�
�H�H�F�B�D�H�M�U�U�V�U�J�H�H�H�H�H�H�H�HÇÃÁÇÊÓßàìù����������ùìàÓÇ�������������������������������������������������
��$�0�=�G�K�F�A�=�9�0�$���m�a�m�o�y�{�������������������������y�m�g�S�A�5�(�"�'�5�A�N�Z�g�s�}�����������g����������
����
���������������������������s�Z�Q�6�8�B�Z�������������������׾A�6�(�!�&�3�A�M�Z�f�s���z�s�f�d�Z�U�M�A�#���!�#�/�:�<�A�<�:�/�#�#�#�#�#�#�#�#�ּ����������������ʼ�����"� ����ֿT�<�'���"�.�;�`�y���������������y�`�T���������������������¾����������������������������������лٻܻ���ܻл��������Z�Y�N�E�D�F�F�H�N�Z�g�s���������~�u�g�Z�f�c�^�c�f�s�����������������s�f�f�f�f�/�%�+�/�;�H�L�H�?�;�/�/�/�/�/�/�/�/�/�/�t�s�g�e�`�g�g�t�t�tìèàÜÜàáìù����������ûùìììì���������������ĿƿĿ��������������������ݿѿ˿οֿݿ�������&�!�������ݹϹùù��������ùϹչܹ�ܹչϹϹϹϹϹϻ:�6�-�+�+�-�7�:�F�G�I�F�A�;�:�:�:�:�:�:���� ���/�<�[�h�{Ċ��n�f�R�Q�B�)�����������ĸıĿ���������
�#�.�0�)�����ƧƝƑƊƄƎƚƧƳ������������������ƳƧ������������������������
���������l�e�_�Y�V�_�l�x���������������x�l�l�l�l�$������"�$�0�:�=�F�A�=�0�)�$�$�$�$���������������������ɺֺֺںֺֺɺ������y�m�U�H�C�B�G�^�m�y�������������������y¿´²²²¿��������¿¿¿¿¿¿¿¿¿¿�������(�2�5�A�M�N�O�S�N�J�A�5�(��N�J�A�?�A�C�N�Z�g�g�n�g�Z�P�N�N�N�N�N�N�r�h�o�r�~���������~�r�r�r�r�r�r�r�r�r�r��
����)�-�)�)���������������������������������������������������ŭţŧŪŭŭŹ��������������Źűŭŭŭŭ��������������������"�)�-�/�+�)����@�?�4�<�@�L�P�Y�c�^�Y�L�@�@�@�@�@�@�@�@�w�n�b�b�[�V�a�m�z���������������������w����"��%�'�4�@�M�M�P�M�G�@�4�3�'���������������������C�P�J�C�6�������ٹ������������ùϹܹ�����������ҹ���������!�'�(�3�@�D�C�@�3�0�'�������!��������!�-�:�S�Z�x�������l�S�:�!�z�y�z�������������������������������z�z����������������'�-�1�2�0�'�������������Ľнݽ�����������ݽнĽ������������~�������������ǽнֽнĽ��������C�:�6�(�������$�*�6�O�Z�a�\�Z�O�CŹŭšřŚŠūŹ����������������������Ź������������(�4�@�A�B�A�8�4�(����S�M�F�B�@�:�F�R�S�W�_�l�p�u�t�r�l�_�S�S����	�� ����!�&�#������������������������������������������������ĿĦģĤħĳĴĺĿ������������������Ŀ�I�A�=�2�=�>�I�V�b�o�v�{ǅǈǈ�{�o�b�V�I�����������$�%�$�"������������������������������������������������������忟�������������ĿϿͿĿ�����������������ED�D�D�EEEE*ECEPE`EiEpEiE\ECE*EEE���������������������
�������
���񾌾������������������ʾоӾ̾ʾ����������ּּѼҼּټ������ּּּּּּּֽ�����������!�.�:�G�K�O�G�?�.�!���(�5�A�N�Z�]�b�\�Z�N�J�A�5�(�"�� �(�(�(D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� 1 A � - 6 & 9 O ^ J c  < N = S E  V R 1 : ! ( ` e [ F J ? 1 1 j . C - M R C X 9 8 Z Z � f ^ p d M : N Y ( , 5 V < 3 Q X 3 { G ; % ` ( % Q    f  �  }  =    :  q  W    �  m  3  �  }  �  \  �  P    �  �     _  �    V    W    �     '  E  b  �  K  H  (  �  �  h  
    �  }  �  ,  �  }  8  �  �  �  k  D  v  A  �  \  �  �  �  |  @  �  �  �  ?  �  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  B>  �  �    D  Z  j  l  e  [  L  2  	  �  �  Q  �  X  �  �  �  A  :  3  ,  #        �  �  �  �  �  �  �  �  �  �  {  ]  N  \  i  {  �  �  �  u  f  X  F  0    �  �  �  �  ~  \  :  /  �  M  �  $  o  �  �  �  �  �  �  �  J  �  �  �  �  �  �  �  �  �  w  e  S  A  /      �  �  �  �  |  W  2    �  �  �  8  �  �  �    7  B  <  ,    �  �  d  �  z  �      �  �  �  �  �  �  �  �  �  �  �  x  o  e  ]  U  N  D  4  #    �    3  B  L  P  M  E  8  $    �  �  }  7  �  �  0    �  �  �  �  �  {  q  ^  B  &    �  �  �  �    a  B  #    �  �  �  �  q  S  0  	  �  �  j  %  �  q    �  q    �     !  �  �  �  �  �  �  �  }  �  t  V  4    �  �  p  E    �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  g  R  8  �  �  �  �    !  "    �  �  �  r  <    �  r    �  K  �  �  E  �  �  �  �      �  �  �  �  j  '  �  �  X    �  T  �  �  �  �  �  �  �  �  �  �  �  �  �    r  b  N  9  %     �   �  �  �  �  �  �  x  a  I  4      �  �  �  �  �  w  [  J  Z  �  �  �  �  w  l  _  R  A  ,    �  �  �  �  e  4  �  �    �  �  �  �  �  �  �  �  �  }  i  S  =  &    �  �  �  �  u  �  �  |  u  o  h  b  [  U  O  K  K  K  L  L  L  L  L  L  L  �  �      "  !      �  �      �  �  �  �  ^  =  Q  Q  �  �  �  �  �  �  �  \  0  �  �  �  ^  &  �  �  v  E    �  e  Z  O  >  *    �  �  �  �  �  �  s  P  .    �  �  �  �              �  �  �  �  �  w  P  $  �  �  u  !  �  "       7  C  H  J  N    �  �  �  q  Z  8    �  �  O    �  I  C  <  6  /  )  #                
           &  M  F  7      �  �  �  {  E    �  ~  4  +  �  x  	  �  �    v  �  �  �  �  �  �  d  &  �  t    �  @  �  Q  �  1  �  }  �  �  �  q  Z  <    �  �  �  Z  %  �  �  `  �  �    �  Y  o  �  �  O  H  >  2  #       �  �  �  �  h  8  �  �  M  �  �  |  \  :    �  �  �  d  I  2       �  �  �  �  �  �      �  �  �  �  �  �  `  >    �  �  �  t  F    �  �  �  C  z  �  �  �  �  �  �  �  �  k  J  !  �  �  �  P    �              �  �  �  �  �  �  �  d  C    �  �  s  I  !    	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  h  �  �  �  �  �  �  �  �  �  �  �  �    h  D    �  �  ^    .  "         �  �  �  �  �  �  �  �  �  �  {  b  I  0    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  o  d  Y  K  >  1  "    �  �  �  �  �  {  p  b  T  F  8  ,          �  �  �  �  �  f  D  :  7  2  "  I  ]  Z  K  5    �  �  �  �  D     �  l  1   �  �  �  �  �  �  �  �  �  �  �  �  �  s  \  C  %    �  �  W    �  �  �  �  �  �  n  n  p  g  R  @  ?  ?  $    �  �  W  /  !      �  �  �  �  �  �  �  �  �  �  l  V  ?  '     �  %    �  �  �  �  �  �  �  q  _  P  A  :  B  K  O  :  &    �  �  �  �    e  F  $  �  �  �  s  3  �  �  F  �  8  i  Y  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  @    �  �  �  �  �  t  Z  H  ,      �  �  �  �  �  t  �  �  Y     �  �  �  �  �  �  �  �  �  �  x  p  g  _  W  O  G  ?  7  /  '  �    %  "    
  �  �  k  L  ;      �  �  e    t  ?  �  -  *  #      �  �  �  �  �  v  N    �  �  G  �  �  :   �  -  (  #  #  +  #    �  �  �  �  t  D    �  �  %  �  <  v  �  �  �  �  �  �  �  �  �    k  U  >  $    �  �  z  4  �  �  �  �  �  �  �  �  �  �  h  E    �  �  q  4  �  �  B  �  �  �  )  C  N  K  ;  $    �  �  �  f    �  ^    �  �  �  �  �  �  �  �  �  s  \  D  *    �  �  �  �  �  ;  �  \  �  E  5  %    �  �  �  �  �  q  U  8    �  �  �  �  e  )  �  g  k  u  �  �  �  w  c  J  ,    �  �  r  2  �  �  {  b  =    *  ,  %      �  �  �  �  �  �  w  Z  "  �  c  �  v  �  �  �  �  �  �  t  c  S  F  4    �  �  �  Y    �  |  -   �  e  f  h  i  i  c  ]  W  O  D  9  /       �  �  �  �  w  O  )  :  A  A  ?  A  M  \  Q  G  ;  -  !    �  �  �  l    �  I  ,    �  �  �  �  �  �  d  D  &  
  �  �  �  �  L    �  ?  D  C  6  %     %          �  �  '  
t  	�  �  F  �  �  D  (  	  �  �  �  }  N    �  �  y  A    �  Y  �    �  &  �  �  �  �  �  ~  j  P  4    �  �  �  �  �  m  C  �  7  [  :      �  �  �  �  r  X  >  #    �  �  �  �  ~  @  �  U  �  �  �  �  �  �  {  q  j  ^  F    �  �  O  �  �    �  �      �  �  �  n  C    �  �  \    �  F  �  w  
  �     b  �  �  �  �  {  g  V  D  4  $      �  �  �  �  �  �  U  �