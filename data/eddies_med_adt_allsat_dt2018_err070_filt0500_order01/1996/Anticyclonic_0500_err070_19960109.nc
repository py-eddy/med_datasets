CDF       
      obs    :   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�^5?|�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N M   max       P��w      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       >+      �  |   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?E�Q�   max       @FB�\(��     	   d   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ҏ\(��    max       @vdQ��     	  )t   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q�           t  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @���          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��o   max       >o��      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�&   max       B1�I      �  4�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��F   max       B1�d      �  5�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C�a      �  6�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�߇   max       C��      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  9P   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  :8   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N M   max       P�7      �  ;    speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�U�=�L   max       ?���,<�      �  <   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �o   max       >+      �  <�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?E�Q�   max       @FB�\(��     	  =�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vdQ��     	  F�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @Q�           t  O�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @�@          �  Pl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  QT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n   max       ?���e���        R<   5         
      �            	   =   L   c      D         c                  c      *      	               O   "       	   )      0   (   .   K      
      �                                  N   O�2#N_p�N�t�O7(ON�"P��wN"hO��DN$�N-�O��PS�Pw�N=2�O�}�N���N'�4P[GaN�qNy�O]�;N�O=��O�/�N��kO��O�Y�NFN�DN��	NZ�N�b�O�w�O�^VOMVO��O��N�dO���O�~�O��P�vO�NL�uN�V�O���O��Nl.�Oe�QNzFFN MN��
N��rNEOA N��O;O�N�,Ѽ����o�D��;o<o<o<o<t�<49X<e`B<u<�o<�C�<�C�<�C�<�t�<���<�9X<�9X<�9X<�j<ě�<���<���<�`B<�`B<�=o=C�=C�=t�=�P=��=�w=�w=#�
=#�
=,1=8Q�=8Q�=D��=H�9=T��=T��=aG�=y�#=�%=�o=��=�+=�O�=���=�{=�E�=�^5=��=��>+TRPU[gt����������g[T����������������������� �����������������������������)5BNNINPTNB5)5Ng���������g81*)������

����������kph���������������|k���������������������������������������������
#/7>AA</#��� ��)5N[iki_5)�����)5N[agh\N)��)+..)!���������������������������������������� &)6BFB<6)          ������"BILMB5���*68@CGIIC:65-*#KGKOQ\^d_\YOKKKKKKKKhktuu������������tphBOV[ehjnhc[OBBBBBBBB	)5BNRZRNB5)	���
/=JPTSH</#
��:7:;CHTahebaWTHHF;:: ,6BOV[^`]OB6) ��~{}~������������������������������������������������������'),567BIHDB6,)''''''�� 
#,%#
�������������������������������)6>CB4������pp����������������up}}�����������������}�������utheedeht���� /<HU]a_]`]UH<63/���������������������
!5;</#	 �����
#'/7973/#
��#/46<EE?/
�|������������������)5BNdo|yj`[2)&#$"��������������������ehmqt�����theeeeeeee��������

����`_a_az����������zmi`����������������������������������������!#&+/<=>A@</%#!!!!!!��������������������
	
##*.'#






��������������������
!!
#-/<B?=</#
hnpz���������zonhhhhnz����������ztnjedhn����������zutwz}����������������������������������������čĚġĦĨĦĠĚĒčĄāčččččččč������������ùõìåìôùÿ�����������ſy�{�������������������y�m�`�P�S�`�m�w�y�.�G�T�`�d�g�g�j�n�l�`�U�G�;�5�,�+�)�-�.āčĥĠėĆ�|�t�[�O�6���������6�\ā�n�o�n�j�n�n�v�n�j�a�\�_�a�k�n�n�n�n�n�n���'�:�3�,�+�'����������������ݿ�����������ݿտݿݿݿݿݿݿݿݿݿ����������������������������������������ҿm�y�����������������y�m�`�T�=�/�;�?�G�m���Ŀѿݿ���@�L�S�S�@�����ݿ���������Ƴ�����������������ƧƁ�u�^�uƄƚƳŹ��������ŹŭţŭŰŹŹŹŹŹŹŹŹŹŹ���ùϹ�����
����ܹϹ���������������ÓàìùùùøìæàÓÇÅÃÇÑÓÓÓÓ�ѿݿ���ݿڿѿ̿ƿѿѿѿѿѿѿѿѿѿ�����0�I�b�b�Z�R�I�<��
���������������񿆿��������y�u�m�`�T�L�G�F�G�T�W�`�m�y����"�.�0�.�$�"��	��	�����������#�-�/�<�B�F�A�<�/�#���
��������
��H�L�U�Y�U�Q�H�<�6�<�A�C�H�H�H�H�H�H�H�H�Z�f�s�s�v�{��z�w�s�f�Z�X�P�J�I�M�N�U�ZECEPE[EKEJEBE7EEED�D�D�D�D�D�EEE*EC�/�;�H�K�T�W�Y�[�T�H�;�9�/�-�"�"�"�-�/�/���ʾξ׾����׾ʾ�����z�������������������(�5�O�a�g�Z�N�����ٿտ������������������������������������������"�"��	��	���"�*�%�"�"�"�"�"�"�"�"�"��� ���������������������������������������������������������M�Z�\�Z�[�[�Z�M�A�@�9�:�A�D�M�M�M�M�M�M�M�P�Y�i�k�b�Y�M�@�4�'������'�4�D�M���������������������s�m�l�_�c�b�g�s������������ĺ������������~�z�o�r�r�~�����F�B�:�-�!���!�$�+�-�:�F�S�^�[�T�S�J�F����������������ùìããîù�������)�5�B�N�[�b�f�[�N�B�5�)�$�'�)�)�)�)�)�)�H�T�a�p�{�z�o�a�/��	�����������	��5�H�������������������������������t�s�{�����׾��������׾Ⱦ������}�p�s�����ʾ�àù��������úõìàÇ�z�l�a�`�nÀËÔà�T�a�h�h�c�Y�;�/��	����������	�"�;�H�T���������������ܻ���������4�7�@�M�X�X�M�@�4�+�)�0�4�4�4�4�4�4�4�4DoD�D�D�D�D�D�D�D�D�D�D�D�D�D{DlDdDbDfDoŭŹ������������������ŹŭŪťŢŚśŠŭE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������ĽֽݽݽнȽƽĽ�������������������¦²´³²¦��"�$�)�&�"�������������������������ž����������������������������S�F�:�8�-�!�!�-�:�F�K�S�_�p�x�~�x�l�_�S�~�������������~�t�s�~�~�~�~�~�~�~�~�~�~���*�6�C�K�H�C�@�5�*���������	����� �"�����������������������������ܻл̻ɻ̻л׻ܻ�����:�D�:�.�!��������������!�.�:�:�:  < V @ . 1 � T O p 7 7 % ] 2 D _ > O W $ � X T ; < v X P * e T C U ) 9 K W Z 0 b C H 4 >  X m Z 4 \  p T $ 4 " s  �  }  �  �  �  �  �  �  L  �  }  �  �  C  �  �  s  �    ,  �  �  �      k  �  Q  8  �  �  �  1  ?  �  @  �  �  ^  (    �  B  `  �  �  F  �  �  �  B  �    W  �  �    	<�����o<T��<49X<�t�>["�<D��=�w<T��<�9X=���=�^5=�<ě�=�{=��<�j=���<�/<ě�=#�
<�/=��=��=,1=�O�=T��='�=�w=#�
=�w=0 �=�h=�t�=�\)=H�9=��
=Y�=�v�=�1=��=��=��=}�=�7L>o��=��=�O�=���=���=�t�=�E�=�v�=\=�F=�F>M��>:^5B	��B�BvKBB 	B	P�B��B|cB�=BswBg�BP�B�EB��B-6B"ZB�VBɨB0�B1�IB2HB�B�B�rA�&B5�B��B&�B#7B�B$�#B"qBB�HB�iB��B�8B�eB
B9�Bi�B~�B
B{B"9�Ba�B��B �uBׯB,�B_(BC%B$�|B-;B��B�tBiB��B�#B	��BD[B��B:B>SB	?�B:�B�oB��B��B�`B@�B��B@�B?�B"@ BEvB�B/��B1�dBBB�B��B�PA��FB@�B:6B?�B#AB�B$A6B"GBŖB�ZB�yB�B�B-�B¥BBAB��B?�B(yB"?�B�B:�B @�B��B,C&BJ�B@�B$��B- 0BP�BDB��B�BŶA���A���A�8pAl7dAe,�A�j�A��@?]�'A[A���Ak7A���Bo`A���>���A�$4A{��A��Aj�pA^2eA�GA�$A@��C�p�A��CAMK�A��qA���A]�5@U�A�dA=1 @ѧ�AF&e@�@}w�A�9A���A���A�`�ANt�A�G�A�U@��M@�4�C��yA�#�C�aA$�A�ےA� AK�@��@�8A���A2��@�X�A>A�|�A�C�A�~�Ak+Ad6A�SAƆ�?batA}��A�w�Ak�A���BT+A�x�>�߇A��A{�A�S�AnA]�A�A��A?��C�v�A�r(AL�eA�zNA��A\��@Wu'A[A<�6@ѥ�AF� @�,@v��A�h�A�N�A�{�A��lAK �Aɍ]A�rB@���@��?C�ȍA�n+C��A$�A��9A�v�AK4�@�f,@LiA�v.A3]@��A�8   5         
      �            	   =   L   c      E         d                  c      *      
               O   "       
   )      1   (   /   L            �                      	            N                     9                  /   3               -                  #         !                  #   #               '      #   '   %            #                                                                                       )                           !                                    '      !      !            #                                 OrN_p�N�t�N�zzO2��O|.pN"hN�
N$�N-�O\�SO��rO�N=2�OQQ�N��\N'�4P�7N�qNy�O]�;N�O!=3Ot�oN��kORD�O�Y�NFN�DN��	NZ�N�b�Oa��O��4OMVO��N���N�dO���ODf�O�}Ol\O�)tNL�uN�V�O5�rO��Nl.�Oe�QNzFFN MN��
N��rNEOA N��O4�N�,�  *  �  �    �  �  �  �  �  p  
�  S  �  �  	  �  �  
�  r  2    �  �  U  d  �  Q  �    �  �  _  
�  l  �  !  m  �  -  R  �  
�  �    X  K  �      �  9  �  �  �  �  �  �  �o�o�D��;��
<t�>%<o<�j<49X<e`B<�=P�`=y�#<�C�=��<���<���=<j<�9X<�9X<�j<ě�<���=D��<�`B=��<�=o=C�=C�=t�=�P=��=<j=�w=#�
=q��=,1=8Q�=]/=L��=��w=]/=T��=aG�>o=�%=�o=��=�+=�O�=���=�{=�E�=�^5=��>+>+Z[`glt���������tg\[Z����������������������� �����������������������������)5BFMNNRNB5)IFGLN[gt������wtg[OI������

����������}z��������������}}}}�������������������������������������������
#/19<<5/#
��)5BNXXUNB5)%
)5BHPRROB5))+..)!���������������������������������������� &)6BFB<6)          �����7@CC=5)����*68@CGIIC:65-*#KGKOQ\^d_\YOKKKKKKKKhktuu������������tphBOV[ehjnhc[OBBBBBBBB
)5BNONNNB5) 


#/6<DILH</#
:7:;CHTahebaWTHHF;::($%)-6BOUY[\[VOB>6)(��~{}~������������������������������������������������������'),567BIHDB6,)''''''�� 
#,%#
��������������������������������")+,(���zy�����������������z}}�����������������}�������utheedeht����*),/<DHRPHG<5/******���������������������
!5;</#	 ����#&/0443.#

#/36<DD>/#��������������������$&$$)5BNbmu{vg[B5)$��������������������ehmqt�����theeeeeeee���������


����`_a_az����������zmi`����������������������������������������!#&+/<=>A@</%#!!!!!!��������������������
	
##*.'#






��������������������
!!
#-/<B?=</#
hnpz���������zonhhhhmhgknvz}��������zxnm����������zutwz}��������������������������������������������čĚġĦĨĦĠĚĒčĄāčččččččč������������ùõìåìôùÿ�����������ſy������������y�m�h�`�Y�\�`�m�n�y�y�y�y�;�G�T�`�a�e�f�f�`�T�G�F�;�7�.�,�+�.�0�;�B�O�[�h�j�r�s�n�h�[�O�B�6�1�)�(�&�+�6�B�n�o�n�j�n�n�v�n�j�a�\�_�a�k�n�n�n�n�n�n�����!��������������������ݿ�����������ݿտݿݿݿݿݿݿݿݿݿ����������������������������������������ҿy�����������������y�m�`�T�J�B�G�I�\�m�y��������(�/�.�(������޿ڿڿݿ��Ƴ��������������������ƧƚƏƉƅƇƚƧƳŹ��������ŹŭţŭŰŹŹŹŹŹŹŹŹŹŹ�ùϹܹ�����������ܹϹù�������������Óàëì÷ìåàÓÇÆÄÇÓÓÓÓÓÓÓ�ѿݿ���ݿڿѿ̿ƿѿѿѿѿѿѿѿѿѿ���#�<�J�M�J�C��
��������������������������������y�u�m�`�T�L�G�F�G�T�W�`�m�y����"�.�0�.�$�"��	��	�����������#�-�/�<�B�F�A�<�/�#���
��������
��H�L�U�Y�U�Q�H�<�6�<�A�C�H�H�H�H�H�H�H�H�Z�f�r�s�u�y�}�x�s�s�p�f�Z�R�K�J�M�O�V�ZEEE*E7E9EAE?E<E7E-EEED�D�D�D�D�D�E�/�;�H�K�T�W�Y�[�T�H�;�9�/�-�"�"�"�-�/�/�����ʾҾ׾߾�ؾʾ����������������������������(�5�O�a�g�Z�N�����ٿտ������������������������������������������"�"��	��	���"�*�%�"�"�"�"�"�"�"�"�"��� ���������������������������������������������������������M�Z�\�Z�[�[�Z�M�A�@�9�:�A�D�M�M�M�M�M�M�4�@�M�Y�]�_�`�\�Y�O�M�@�4�'�%����'�4�����������������������s�f�f�h�g�j�s������������ĺ������������~�z�o�r�r�~�����F�B�:�-�!���!�$�+�-�:�F�S�^�[�T�S�J�F������������������������������������)�5�B�N�[�b�f�[�N�B�5�)�$�'�)�)�)�)�)�)�H�T�a�p�{�z�o�a�/��	�����������	��5�H�������������������������������}�y����������������׾ž������~�w�t������ʾ�ÓàìññìãàÓÇ�z�y�p�l�m�r�zÇÐÓ�"�;�H�T�a�d�e�`�W�;�/�"��	����������"���������������ܻ���������4�7�@�M�X�X�M�@�4�+�)�0�4�4�4�4�4�4�4�4D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DzDrDsD{D�ŭŹ������������������ŹŭŪťŢŚśŠŭE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������ĽֽݽݽнȽƽĽ�������������������¦²´³²¦��"�$�)�&�"�������������������������ž����������������������������S�F�:�8�-�!�!�-�:�F�K�S�_�p�x�~�x�l�_�S�~�������������~�t�s�~�~�~�~�~�~�~�~�~�~���*�6�C�K�H�C�@�5�*���������	����� �"��������������������ܻ��������	� ����ܻлϻ˻λлۻܽ:�D�:�.�!��������������!�.�:�:�:  < V 4 !  � 7 O p 2   ] ' @ _ 9 O W $ � Q 3 ; 9 v X P * e T . : ) 9 ! W Z & a  C 4 >  X m Z 4 \  p T $ 4 ! s  �  }  �  �  �  �  �  �  L  �  �  �  �  C  �  �  s  �    ,  �  �  �  �    �  �  Q  8  �  �  �  �  ?  �  @  �  �  ^  �    �    `  �  {  F  �  �  �  B  �    W  �  �  Z  	  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  g  �  �  �      &  (      �  �  �  m    �  �    [  �  �  �  �  �  �  �  �  �  �  �  x  o  f  ^  U  '  �  �  x  =  �  �  �  o  Z  C  1      �  �  �  �  j  I  )  	  �  �  �  �  �                    �  �  �  �  �  n  J  (    �  �  �  �  �  �  �  �  �  u  c  Q  <  %  �  �  �  �  �  �  	�  
�  �  �  w  "  �  e  �  _  �  �  �  *  y  C  �  
�  �  �  �  �  �  �  �  	    �  �  �  �  �  �  �  �  �  �  �  �  {  �  �  �    h  �  �  �  �  �  �  �  �  �  t  :    �  �  �  �  �  �  y  q  i  a  Y  P  H  C  @  >  ;  8  6  3  1  .  +  p  Z  D  A  J  N  @  2    �  �  �  �  q  L  %  �  �  �  �  
  
K  
}  
�  
�  
�  
�  
l  
1  	�  	�  	Q  �  x  �  5    �  �  O     f  �  0  �  �  
  4  K  S  A    �  �  N  �  d  �  M    �  s  �    Q  �  �  �  �  �  �  �  <  �  �    k  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  c    �  �  q  O    �  �  �  �  	   	  	  �  �  �  �  H  �  �  �  6  L  #  K  �  �  �  �  v  W  3  
  �  �  ]    �    $  �  %  �     x  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  a  :     �  	�  
  
b  
�  
�  
�  
�  
�  
T  
  	�  	�  	A  �  n  �  I  T    {  r  j  a  Y  P  G  ?  6  ,            �   �   �      ,  A  2  +  %                �   �   �   �   �   �   �   �   �   �   �      �  �  �  o  f  \  L  5    V  Q  7    �  �  �  J  �  �  �  �  �  �  �  �  �  �          "  (  -  3  8  >  C  �  �  �  �  �  �  s  `  L  9  %    �  �  �  �  �  y  U  (  2  �  �  7  Q  T  B    �  �  /  �  9  �    
@  	0  �  �  ?  d  S  D  :  9  &    �  �  �  �  j  A    �  �  =  �  �  .  *  a  x  �  �  �  �  s  W  /    �  �  Z    �  9  �  �   �  Q  5    �  �  �  k  3  �  �  m  !  �  y  4  �  h  �  .   b  �  �  �  �  �  �  n  N  '    �  �  �  �  �  m  Q  5     �        �  �  	    +  6  7  8  9  ;  >  A  D    �  j    �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  a  E    �  z  �  �  r  _  M  :  '                
       �   �   �  _  X  P  I  @  6  +      �  �  �  �  �  �  q  X  5     �  �  	�  
  
N  
�  
�  
�  
�  
�  
~  
N  
   	�  	  v  �  �  �  �  �  7  A  X  f  l  f  X  C  "  �  �  �  ]    �  R  �  z  '  �  �  b  B    �  �  �  m  G    �  �  y  .  �  �  ,  �  _  �  !    �  �  �  �  S    �  �  �  �  �  �  �  �  �  �  q  J  Z  d  ]  U  �     B  V  f  l  j  ^  F    �  �    �    �  �  s  c  S  C  3  $      �  �  �  �  �  �  �  �  v  ]  ?  -  )    �  �  �  ^  5  !  �  �  �  A  �  i  �  C  �  5    �    *  C  Q  R  K  =  (    �  �  <  �  w  �  t  �    �  �  �  �  y  ]  =    �  �  d  7  �  �    �  j  �  v  �  n  	$  	Y  	�  	�  
	  
V  
�  
�  
�  
d  
,  	�  	�  	  �  �  �  x  0  �  �  �  �  �  �  �  �  �  S    �  �  �  �  �  q  0  �    .    �  �  �  �  �  u  _  I  2      �  �  �  �  �  [    �  X  K  >  1  '        �  �  �  �  �       
      �  �  �  G  )  �  �  �  /  I  ;    �  g  �  �  \  �        
  �  �  �  �  �  k  I    �  �  {  4  �  �    �  h  �  �  q            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  |  ]  ;    �  �  l  )  �  �    �  �  �  �  �  �  r  Z  L  J  F  @  !  �  �  �  {  d  I  -  9  3  -  '  !            �  �  �  �  �  �  �  �  �  �  �  �  r  W  +  �  �  �  ~  >  �  �  f    �  {  :    �  �  �  �  s  a  O  =  *      �  �  �  �  �  �  �  �  �  �  �  �  �  |  r  i  _  V  L  A  6  *      �  �  �  �  C  �  �  �  u  Y  0    �  �  �  �  ]  6    �  �  �  M    �  �  �  �  �  �  �  �  i  B  $    �  �  �  �    �  �  �  �  �  �  t  ~  �  a  )    �  �  t     �  W  �  4  y  
�  	�  /  �  �  �  �  �  �  p  K    �  �  �  �  j  >    �  �  {  G    �