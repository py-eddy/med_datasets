CDF       
      obs    :   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�z�G�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�6�   max       P�>�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �8Q�   max       >��      �  |   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @E�
=p��     	   d   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       @��
=p�   max       @v�=p��
     	  )t   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q@           t  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ȝ        max       @��          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >���      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B,�      �  4�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��#   max       B,84      �  5�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       @��   max       C��      �  6�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @�<   max       C��p      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  9P   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  :8   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�6�   max       P?%�      �  ;    speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�XbM��   max       ?�d��7��      �  <   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �8Q�   max       >��      �  <�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @E�
=p��     	  =�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       @��
=p�   max       @v�=p��
     	  F�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q@           t  O�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ȝ        max       @�X`          �  Pl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�      �  QT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��`A�7L   max       ?�c�A [�        R<      
               q                                        �      a   �            D         ?                  	         	      A      1                     
   .            	   ]      NObwN���Nb�N��Ot�N.�aP�>�OJyOrl�OS8�O�O(�YOJ�Oq�Nv�hN�O���N�9�O�P���Nճ"P�[P��N�}.O���Of��PH�~O��O�n~P1R�On��O5��P?%�N���N�UN�Oz�Nb.N��AN���P+ O�gwP,��N0
Nߑ�N�٪N��YM�6�O;fN��O��3NsH�Nǻ�N���N=��O��WN���O&1ؽ8Q켛�㼛��ě���o�o�o��o:�o;��
;��
<o<o<o<o<49X<D��<T��<e`B<e`B<e`B<�o<���<���<���<���<�/<�`B<�h<�h<��=+=t�=�w=�w='�='�=49X=8Q�=<j=@�=@�=D��=L��=T��=T��=aG�=aG�=u=�o=�hs=��w=��w=ȴ9=�x�=�x�>��>��'%*/<E@<6/''''''''''GDA?HJUafa][UPJHGGGG�����������������������	�������������������������������������������������������)5BHFLKD5)����!)6>BEMMHB6) ��������� ����������&)5BNSQ[baRB5)mmu����������������m�	"/7;@CJNHC;7/	�~|~���������������2<BN[gkty|}tgb[NA:62�������������������������������������������
#$"!"
�������������������������
 8HUXY[U@</#
���������
�������#$0<IRSQI<<0+#�������������������������
!%&$
������������  ��������������

�����kdenz�������������zk)B\fnoje[N5)!	���
:@KXaUH;/%����������!(()&����2++3B[t��������th]I2BINS\gt��������tg[NB���%)-486)&������3BQV\[N5)����������� 
������	
#.0520(#
626<HIKHC<6666666666��������������������#/83/#�����(&������@@BGO[dhhhe\[OJB@@@@��������
���������������������������������)5?DGB)����)*,*)���������������������)*,*) ��).24)��������������������2/8;DHTabmmpmjaTH;22<:<HMSOHE<<<<<<<<<<<��������������������LJN[gtt}tg[NLLLLLLLL���������������������

���������afnz��zqnmaaaaaaaaaa�����������
��������������������������)6AB?6)(�D�EEEEEED�D�D�D�D�D�D�D�D�D�D�D�D����Ŀѿݿ������ݿѿǿĿ���������������������������������������������������ù������������ùìãìôùùùùùùùù�5�B�N�R�W�Y�W�N�B�5�)�$��!�)�3�5�5�5�5ŠŭŹ��������ŹŭūŠŜŠŠŠŠŠŠŠŠ�0�nŇŔŖŚŘŇ�n�I�
������ĽĿ������0�������	��� �!���	�����������������仪�ûʻлû��������x�p�j�r�x��|���������.�;�G�T�`�c�i�h�i�`�T�K�G�;�.�(�*�*�,�.�G�T�`�h�n�p�q�e�`�T�G�;�.�"� ���"�.�G�T�a�g�o�t�r�m�h�a�T�H�;�-�/�3�;�H�N�R�T�A�N�Z�g�s�}�x�s�i�g�Z�N�A�@�5�2�5�8�A�A����������������������������������������F=FJFVFWF^FVFJF>F=F1F.F1F5F1F=F=F=F=F=F=�� �"�&�"��	�������	�����������������,�8�8�5�/�����������������޾f�s��������s�f�e�Z�V�Z�[�f�f�f�f�f�f�a�m�l�r�z���������m�a�H�/�&��'�>�?�G�a������������������s�N�6�5�@�B�4�5�s���������������������ۺֺֺԺֺں����� �#�� ����ì�z�V�D�?�G�U�nÇàù����� DoD{D�D�D�D�D�D�D�D�D�D�D�D�D{D_DUDVDbDo�������
������
��������������������(�4�A�H�M�Q�]�f�]�M�A�4��������(�tāčĚĢĥĦĨĬĳĳĦġėč��t�n�l�t��A�[�c�a�P�5�(����ݿϿÿ����Ŀ������������y�m�T�;�"���.�;�G�T�m�y�������;�G�T�`�d�e�a�X�T�G�;�.�"��	����.�;�s�����������������Z�M�E�I�D�F�H�M�Z�s�"�/�;�H�T�^�_�Z�T�P�H�;�/�"������"���-�6�4�*�!��������������������������������ƳƚƎƃ�~ƀ�~ƑƧƳ�����������������������������{�}�����ʼּڼ���������ּӼʼż��ļʼʼʼ����������������������������������������޽������������������������������{�z�}�����ݽ�������ݽؽҽֽݽݽݽݽݽݽݽݽݽݾf�s�s�s�n�i�i�f�f�Z�M�E�C�E�I�M�Z�]�f�f�ѿݿ�������߿ݿܿѿĿĿ��ĿͿѿѿѿ��s�������������������������o�Z�I�A�N�f�s�ѿ�����5�%���	����ݿѿ��������Ŀ���������������
������²®­²³¿�������������������������������������������"�/�;�H�T�`�Z�T�H�;�3�/�"�"�"�"�"��"�"�� �"�#�#�"���	���������������	����H�U�[�a�f�i�f�a�U�H�C�A�?�F�H�H�H�H�H�H�<�H�U�Y�U�K�H�<�8�<�<�<�<�<�<�<�<�<�<�<��#�0�3�<�?�D�>�<�0�.�#��������F=FJFRFSFJF=F1F.F1F=F=F=F=F=F=F=F=F=F=F=�l�y�����������������������y�r�]�[�`�`�l�[�h�l�t�o�h�h�[�Z�V�W�Y�[�[�[�[�[�[�[�[�[�\�]�[�V�O�C�B�6�)��'�)�6�6�B�O�U�[�[ǮǱǭǧǡǔǈ�{�z�v�{ǃǈǔǡǮǮǮǮǮ������������������������EPE\EiEuE�E�E�E�E�E�E�E�E�E�EuEiEPEIEGEP�лܻ��������������������ܻллͻм�� �'�9�@�M�T�V�M�@�4�'������� N d z S ! Y D % f . 7 n B % D c O : G 8 O G ) / . | 0 � , Q 6 T * c ; : 6 B a  9 [ 6 N @ 4 C S 9 N 0 b r / N @ � c  l  �  ~  K  =  j  T  �  T  �  v  �  3  �  �  K  �  |    �      �    *  x  d  �    M  �  �  ;  �  �  (  +  �  V  �  �  M    K  	  �  �   �  c  '  )  �    �  h  �  5  ��o�#�
��o;o<t�;o=�S�<���<���<�t�<��<��
<���=\)<e`B<u=49X<�o=t�>/�<���=�`B>���=D��=0 �=@�=��=T��=P�`=�^5=u=H�9=�7L=P�`=H�9=H�9=]/=@�=Y�=u=�S�=��=ě�=]/=�%=}�=�hs=q��=��T=��P=�h=�E�=ě�=ȴ9=��m>R�>)��>8Q�Bc�B��B!��B�zB��Bk�B�B�%B"�B�rBޕA���B ��BűB/B/SB�FB�+B`GBv|B& �BZBB��B#�jB ��BT�B�$B��B�=B	��B�B�}B#�CB%YB�$B��B�]BT8B�BʅBBY4BpB��B�B�NBz�A�0;B9!B,�B��B�zB��B�:Bk�B-�B�aBB0B�qB!�'B�B�jB�BB3�B�)B!ТB:B�GA��#B ��B�iB��B:�BI�B�ZB�7BA�B%�wBA�B<�B�mB#�>B �OBcqBu%B� B�dB	�mBVBS�B#�FB%@B��B��B�fB��B��B nBDjB�B{hB�yB<�B�VBBEA�}�B?�B,84B�MB��B��BɁB�B�B@:C�[A{_A@��A�ųA�a�A�LA�A�ݰ@��JAd��AeE�A�7A�.WA���C��A\�A�e�AB`A���A��@J��A�>�C��oA��A9�A޶A��Ag�Ac	tAD�A�U�@^��B:@���A�SA�y_A lUA+�A?�xA|�RA��vA��A�y�A��wA���A���A�;4A�x�A꜕C���AȺA��]A�^�BB�@N��C�
@� �@�a�C�XA|�@�<A�|4A��A��A��A���@���Ad��Ae�A��iA��2A��C��pA\�qA�~�AB��A�UaA��@K�-A�niC���A�WA7�AޤTA���Aj�GAc�AD�TA�lY@dR�BN�@�CA�EAь�A"��A+��A?&aA{JoA���A}��A�� A��'A���A���Ań�Aā�AꄈC��A��A�}A׮B6�@LJ�C�@�@�t      
      	         r                              !         �      a   �            D         ?                   	         	      A      1                        .            
   ]                           7      %                              '   =      ;   '            -   +      +         -                        +      )                                                                  +                                    !   '                        %               -                        '      )                                             NObwN���Nb�N��N�p�N.�aP)��O��O%
�OS8�O��YNx N��N,!Nv�hN�Ok5N�9�O���P�kNճ"OȃOb�NÐ�O���O	|qO��O��VO�n~O��WOLmNO&d"P?%�N�RN��zN�Oz�Nb.N��AN���P0TO�gwP,��N0
N��ON�٪N��YM�6�O;fN��O��3NsH�N���N���N=��OvrN���O&1�  K  P  �  �  '  u  	  ~  d  B  (  �  �  *    ~  Q  �    �  �  �  �    $  5  �  F  D  7  �  #    �    �    B  G  e  �  �  �  	  (  �  R    �  �  I    �    �  �    ̽8Q켛�㼛��ě�%   �o=\);ě�;��
;��
<t�<T��<t�<���<o<49X<���<T��<���=��T<e`B=�o>1'<�`B<���<��=e`B<�h<�h=Y�=C�=C�=t�=,1=#�
='�='�=49X=8Q�=<j=Y�=@�=D��=L��=Y�=T��=aG�=aG�=u=�o=�hs=��w=��
=ȴ9=�x�>�P>��>��'%*/<E@<6/''''''''''GDA?HJUafa][UPJHGGGG�����������������������	������������������������������������������������������,1;@=5)���)! ()68ABFIHBB6)))��������������������&)5BNSQ[baRB5)ss{���������������{s"*/98/"�}����������������KNX[gimgf[TNKKKKKKKK������������������������������������������

������������������������#)<HKPRRVSNH</#
��������������������#$0<IRSQI<<0+#����������������������������

�����������������������������

�����qqwz�������������zqq)5BNW]__[NB5)����
#/9@JW`UH<0'������!(()&����JP[ht����������th]UJLNU[_gt�������tg[ONL��$)-366)' ������3BQV\[N5)���������������������

#,0400&#




626<HIKHC<6666666666��������������������#/83/#�����(&������@@BGO[dhhhe\[OJB@@@@���������	���������������������������������)5?DGB)����)*,*)���������������������)*,*) ��).24)��������������������2/8;DHTabmmpmjaTH;22<:<HMSOHE<<<<<<<<<<<��������������������LJN[gtt}tg[NLLLLLLLL���������������������

���������afnz��zqnmaaaaaaaaaa��������������������������������������)6AB?6)(�D�EEEEEED�D�D�D�D�D�D�D�D�D�D�D�D����Ŀѿݿ������ݿѿǿĿ���������������������������������������������������ù������������ùìãìôùùùùùùùù�B�N�O�S�S�P�N�B�5�.�)�'�)�*�5�?�B�B�B�BŠŭŹ��������ŹŭūŠŜŠŠŠŠŠŠŠŠ�#�0�I�[�r�{�{�n�U�0�
�����������������#���������	�����	������������������׻Ļ̻û��������������������������������Ŀ.�;�G�T�`�c�i�h�i�`�T�K�G�;�.�(�*�*�,�.�G�T�`�d�j�l�m�f�`�T�G�;�6�+�#�!�'�3�;�G�T�a�i�m�o�m�l�a�T�Q�Q�T�T�T�T�T�T�T�T�T�A�N�Z�g�s�z�t�s�g�f�Z�N�B�A�5�4�5�:�A�A����������������������������������������F=FJFVFWF^FVFJF>F=F1F.F1F5F1F=F=F=F=F=F=�� �"�&�"��	�������	��������������� �+�)�$�����������������������f�s��������s�f�e�Z�V�Z�[�f�f�f�f�f�f�a�l�v�z���������z�m�a�T�H�A�6�/�9�H�R�a�������������������������o�h�e�h�s�����������������������ۺֺֺԺֺں����Óìù��������������ùìàÇ�z�q�n�s�ÓD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DuDvD{D�D��������
����
������������������������(�4�A�H�M�Q�]�f�]�M�A�4��������(āčĚĜġģĤĤĚččĂā�{�t�t�t�xāā�����(�6�>�<�3�'�������ۿտ׿����m�y�������������y�m�T�;�"���.�;�G�T�m�;�G�T�`�d�e�a�X�T�G�;�.�"��	����.�;��������������������s�f�_�W�V�Y�Y�f�s���/�3�H�O�T�[�]�X�M�H�;�/�"�!����"�$�/��-�5�3�)�!����������������������������������ƳƚƎƃ�~ƀ�~ƑƧƳ��������������������~��������������������ּ׼��������ּռʼǼüʼѼּּּ����������������������������������������޽������������������������������{�z�}�����ݽ�������ݽؽҽֽݽݽݽݽݽݽݽݽݽݾf�s�s�s�n�i�i�f�f�Z�M�E�C�E�I�M�Z�]�f�f�ѿݿ�������߿ݿܿѿĿĿ��ĿͿѿѿѿ��s�������������������������r�g�Z�R�N�Z�s�ѿ�����5�%���	����ݿѿ��������Ŀ���������������
������²®­²³¿�������������������������������������������"�/�;�H�T�^�X�T�H�;�/�%�"�!�"�"�"�"�"�"�� �"�#�#�"���	���������������	����H�U�[�a�f�i�f�a�U�H�C�A�?�F�H�H�H�H�H�H�<�H�U�Y�U�K�H�<�8�<�<�<�<�<�<�<�<�<�<�<��#�0�3�<�?�D�>�<�0�.�#��������F=FJFRFSFJF=F1F.F1F=F=F=F=F=F=F=F=F=F=F=�l�y�����������������������y�r�]�[�`�`�l�[�h�l�t�o�h�h�[�Z�V�W�Y�[�[�[�[�[�[�[�[�B�O�[�[�[�T�O�B�A�6�)�!�(�)�6�9�B�B�B�BǮǱǭǧǡǔǈ�{�z�v�{ǃǈǔǡǮǮǮǮǮ������������������������EuE�E�E�E�E�E�E�E�E�E�E�E�E{EuEqEjEmEuEu�лܻ��������������������ܻллͻм�� �'�9�@�M�T�V�M�@�4�'������� N d z S - Y I + V . 3 T A E D c F : ; % O 3 # - . Z ' � , : . P * 0 1 : 6 B a  8 [ 6 N ; 4 C S 9 N 0 b l / N  � c  l  �  ~  K  �  j  ,  (  �  �    �    V  �  K    |  6  7    �  �  �  *  U  �  N    X  �  �  ;  .  �  (  +  �  V  �  o  M    K  �  �  �   �  c  '  )  �  �  �  h  Q  5  �  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  K  4      �  �  �  �  p  F    �  p  $  �  �  4  �  �  3  P  C  6  ,  "        !  2  L  c  v  �  �  �  �  �  �    �  �  �  �  �  �  |  r  g  [  P  E  9  *    �  �  �  �  �  �  �  �  ~  x  s  t  t  u  v  z  �  �  �  �  �    P  �  �  	        !  %  '  &         �  �  �  b  #  �  �  m  P  u  o  i  d  ^  Y  U  Q  M  I  G  F  F  F  F  F  E  D  D  C  �  �  Z  �  �  	  	  	  	  	  �  �  �  o     �  '    �  �    H  c  q  {  ~  x  m  [  C  %  �  �  �  J  �  �    �  �  �  F  [  c  d  d  a  Z  M  9    �  �  �  �  �  �  c  4  �  B  =  9  7  0  #       �  �  �  �  i  ;    �  �  �  M   �      !  (  %        �  �  �  �  �  �  _    �  f  �  \  �  �  �  v  z  �  �  �  �  �  z  S    �  �  L  �  �  C   �  �  �  �  �  �  �  �  n  X  @  "    �  �  �  S    �  �  O  �  �  �  �  �  �           $  '  )  (    �  W  �  F  W                �  �  �  �  �  �  �  �  �  �  �  �    ~  x  r  k  e  _  Y  R  L  F  E  I  L  P  T  ^  i  u  �  �     )  >  K  P  O  H  ?  -    �  �  l  &  �  �  d  .    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  t  n  h  b  \  �                �  �  �  �  �  �  �  �  �  w  V  f  	>  
	  
�  .  w  �  �  �  �  �  �  k    
w  	�  �  �  f  �   �  �  �  �  �  u  j  [  L  8  "  	  �  �  �  �  g  6  �  �  �  �  8  �  �  E  w  �  �  �  �  �  �  �  9  �  *  m  u  1  �  �  �  �  �  L  �  _  �  �  �  �  j  �  �  �  r  y  R  �  	l  �  
          �  �  �  �  �  ^    �  z    �  Q  �  Z  $    �  �  �  �  o  D    �  �  s  >    �  �  k  5      *  '  !      *    �  �  �  h  7  �  �  m    �  r  6  �  �  �    "  @  Z  r  �  �  �  x  O    �  G  �  �    :  �  �  @  %    �  �  �  ^  G    �  �  �  Y    �  _    Z    D  ;  2  '      �  �  �  �  �  l  I  !  �  �  }  >  �  �  �  �  �  �  �  �    5  3  !    �  �  p    �  *  �  �  -  �  �  �  �  �  �  �  d  1  �  �  d    �  u    �  0  �  x           �  �  �  �  �  p  I    �  �  |  H    �  �  %    �  �  �  �  v  _  P  J  B  :  0    �  �  �  o    �   �  D  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  p              �  �  �  �  �  a  ;    �  �  �  �  W    �  �  �  �  ~  t  _  K  .    �  �  �    T  (  �  �  �  q    �  �  �  �  �  �  �  n  Z  @    �  �  �  �  �  d  F  (  B  @  =  ;  8  6  3  /  +  '  #                    G  *    �  �  �  �  �  �  �  �  �  ~  l  Y  B  (  
  �  �  e  d  _  V  F  -      �  �  �  �    b  C  !  �  �  g  �  %  �  �  x  _  ;    �  �  �  o  9  �  �  f  �  ;  �  �  h  �  �  �  �  �  h  ;    �  �  �  e  (  �  �  �  �  p  `  L  �  �  �  �  w  F    �  �  �  W    �  �  �  o    d  �  `  	    �  �  �  �  �  �  �  �  �  �  �  �    q  c  T  E  7    #  "    �  �  �  �  �  k  I  '    �  �  �  c  '  �  [  �  �  �  �  �  �  �    h  M  0    �  �  �  �  }  d    �  R  F  6  #    �  �  �  �  �  �  z  H  �  R  �  �  0  �  `      �  �  �  �  �  �  �  �  �  �  y  c  N  9  %    �  �  �  �  �  �  �  �  �  o  O  *  �  �  �  H    �  �  �  o  L  �  �  o  T  6    �  �  �  I  
  �  �  H    �  {  5   �   �  I  @  9  =  >  <  9  9  :  7  (    �  �  �  B  �     �      �  �  �  �  q  I    �  �  �  p  C    �  p    �  `    �  �  �  �  �  �  m  H  "    �  �  �  j  '  �  �  9  �  �                                          �  �  �  �  ~  ]  =    �  �  �  �  �  z  n  f  m  u    �  l  Q  H  |  �  �  �  �  �  �  v    w  �  �  
�  	�  S  o  �      �  �  �  �  �  �  e  8  	  �  �  u  B    �  �  �  �  �  �  V  4    �  �  D  �  �  �  l    �  )  �  U  �  �   