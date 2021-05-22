CDF       
      obs    L   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�dZ�1     0  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�?�   max       Pa�r     0  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �Ƨ�   max       <���     0      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��R   max       @FǮz�H     �  !<   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ۅ�Q�    max       @vU\(�     �  -   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q�           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @��         0  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��-   max       <�o     0  :�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��6   max       B4��     0  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B4�L     0  =$   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?/�   max       C�l�     0  >T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?<��   max       C�i[     0  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          k     0  @�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;     0  A�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3     0  C   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�?�   max       P6F\     0  DD   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�֡a��f   max       ?�c�e��O     0  Et   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �Ƨ�   max       <���     0  F�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��R   max       @F��G�{     �  G�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ۅ�Q�    max       @vU�Q�     �  S�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q�           �  _�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @�0@         0  `,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D   max         D     0  a\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Z��   max       ?�c�e��O     �  b�            
                     $         >      &      -      #      	      ,   !         	               
            k   	               P   
       >            	      
   	   
   	      e               &   M   #         	         2      2         %   N9/OW�N���N�E�N��vN7iNJ�'M�?�N�
!N5*,O���N<[�N��pP20�N��_O�lOH��P[uN��O�N�OX�8N�ތN��2O�.�O�}�O0wM��$N�X�O�tN�[�O�z�N�"�N��NE8�O�KgM�hP$�'N�;O�\%N�h�O�@�O wHPa�rNLgOi�BO��Nkj�Ne�JO_�'O 1O@�?N���O�ٖN�y�N0�O�V7P7lDO FiN2�JNbN꤈P6F\P8�aOe�SO_�O���N�6�N���O=�0O�c�N���O�M�NO EN�^N�^hNx`<���<D��<o;�`B;ě�;�o;D��;o%   �#�
�#�
�e`B�u��o��o��C���C���t���t����㼼j��j��j���ͼ�������������/��`B��h��h�����o�+�+�C��C��C��\)�\)�t��t���P��P����w�#�
�0 Ž49X�8Q�<j�D���H�9�T���T���Y��]/�]/�]/�]/�ixսm�h�m�h�q����%��o������������T���T���T�����Ƨ�:<HUX`VUSHA<::::::::��������������������)/;;BF;9/&#&))))))))����������������������
 
��������,/6<AHIHC</,,,,,,,,,##/18//'#����������������������������������������"' ��
#HannaVO</+�������

������������!#/4:5:9/#  !!!!!!@BP[gt���������g[D@@
#/1<A<81/.#
��������������������
"*6CEHHE?6*
Ham�����������zaTHDH��������������������)0B[tzzwwthO@8674)���������������#()5BKIBCB5))57BNRONHB5.)lmqz������������ztnl=BN[gt���������tO@==��������������������JOR[hihc[UOLJJJJJJJJ ! ���������������������./6;=@@>;/.+)+......6;HTawyxr_WTH;896236��������������������#/0:;20#").6766)'��������������������ghity{thecgggggggggg������ ("���������36BEOOTROB?6-*333333�����
'#
�������������������������������������������}}����������������������0Ib{�{yqbUIB0#������������������������������������������#/<HRT</
�������������������������qtw�������tsqqqqqqqqMN[gt}����|vtg[NLJIM��������������������U[gt�������kg[WSQQU�)-21.)"���:CIUbnzzwof_RI<3257:MO[hpsnlmhc[YPOIMMMM��������������������nr{�����������{nkinn����	����������uz������������zrouu�����������������������������������������������������������������)02.)��������
#<HUmqmaH;�����
#'/89973/#����������������������������������������nnz�������}znddfnnnn������������������������
!��������)5NW\fdkh[VN5)qtv���������vtjlqqqq����������������wx{���������������������zz|���������zzvzzzz<@A?</(#!#/<<<<<<<���������������������4�0�,�/�4�A�M�Q�M�I�A�6�4�4�4�4�4�4�4�4�������ʾ̾˾ʾž�����������������������ĳĮĲĳĿ������������Ŀĳĳĳĳĳĳĳĳ�!���������!�-�:�>�D�:�-�%�!�!�!�!�I�G�I�J�R�V�b�o�{�~�~�{�p�o�b�V�I�I�I�I�$������$�'�0�6�7�0�$�$�$�$�$�$�$�$�;�9�/�"�"�!�"�/�:�;�H�K�H�?�;�;�;�;�;�;�/�-�#��#�/�<�<�H�R�H�<�/�/�/�/�/�/�/�/��x�~��������������������������������0�0�0�0�7�=�I�K�K�I�@�=�0�0�0�0�0�0�0�0����������!����&�B�O�h�f�[�B�)�������������������������������������������H�C�A�H�U�a�n�zÇÆ�z�n�a�U�H�H�H�H�H�H�H�;�����������/�H�a�m�x�v�����m�a�HìêàÓÓÊÎÓàìóù��������ù÷ïì��q�f�Y�@�4�'����1�A�Y�r�~����������������۾޾����	��"�+�.�*�"��	��������������	��/�H�T�g�l�^�b�d�b�U����A�;�4�4�4�5�A�M�Z�`�f�i�p�f�Z�M�A�A�A�A�����������پ�	��.�4�8�7�.����ʾ������N�F�F�M�N�Z�^�g�s�y�������������s�g�Z�N�Z�Y�N�A�@�:�9�A�N�Z�g�j�i�o�o�g�Z�Z�Z�Z���������������������������������������������������������������$�&�(�&������������������	��"�*�,�)��	���𾱾������������ʾ׾����������׾ʾ����Y�W�Y�]�d�e�j�r�t�r�i�e�Y�Y�Y�Y�Y�Y�Y�Y��������!�-�7�4�-�*�!�������;�6�.�*�*�.�4�;�G�T�`�m�s�w�m�b�`�T�G�;�����������������������������������������������(�5�A�Z�g�m�{�|�s�g�Z�5�(��ݽѽнŽȽнݽ����������ݽݽݽݽݽݻû����ûŻлܻ����ܻл˻ûûûûû�ìçéì÷ù��������ÿùìììììììì�������������*�6�C�\�h�s�h�O�E�6�%��������'�,�)�'������������:�-�)�-�S�_�j�l�x�������ĻŻ������x�_�:�_�]�]�_�k�l�x���������}�x�l�_�_�_�_�_�_�ɺ����������ɺ��!�1�+�5�9�+�!���ֺ�ƧơƣƧƳ��������ƳƨƧƧƧƧƧƧƧƧƧÓÍÇÃÇÏÓë����������	�������àÓ�L�K�K�N�R�Y�b�e�r�|�~���������~�r�e�Y�L�����v�}�����Ľ��A�N�H�8�(��ӽɽ������)�&�)�1�6�B�O�Q�O�B�=�6�)�)�)�)�)�)�)�)�[�X�]�nāčėĦĳľ����ĿĹĦčā�t�g�[D�D�D�D�D�D�D�D�EEE*E1ECEHEGE>E*EED�Ň�}ŀŇőŔŗŠŢťŠŔŇŇŇŇŇŇŇŇ�<�3�/�'�%�/�<�F�H�R�K�H�<�<�<�<�<�<�<�<����������������)�5�B�B�:�4� �����ŭŪŠŔŎŔşŠŭŹ��������������Źŭŭ�6�/�.�/�7�=�>�C�E�H�O�\�u�{�q�h�\�O�C�6�T�P�H�H�F�D�H�T�a�m�u�z�|�z�t�m�l�a�T�T���s�n�h�d�f�s������������������������������������������������������������������������$�&�+�$����������������|���������л���������ݻл������f�M�@�9�@�M�r�����ּ�������ּ�����f�����������������
������
����������0�-�%�0�<�G�I�K�P�I�A�<�0�0�0�0�0�0�0�0������������������������ààÖÕÓÓÓàìôù����������ùìàà�y�`�G�6�;�`���������������ݽн��y�ѿĿ������������Ŀѿ޿���A�L�(����ݿ������s�g�[�a�g�o�����������������������������������(�5�6�<�:�7�5�(���H�;�2�2�7�;�?�H�T�a�j�m�w�������z�Y�T�H���������������������ʾ̾ξʾ������������r�o�r�x�����������������������r�r�r�r�����ļʼ׼���������������ּ��z�n�b�]�b�nÇÓàìùþ��������ìàÓ�z�#�����#�$�/�<�H�J�I�H�H�<�/�#�#�#�#�y�o�l�m�t²�����������������²ā�}�t�q�tāčĚĥĚđčāāāāāāāā�ܹ۹ܹݹ���������������ܹܹܹ�D{D�D�D�D�D�D�D�D�D�D{DpDqD{D{D{D{D{D{D{�������#�0�5�8�0�#�������� W ' 0 H < 0 Q � t Z g T e % * L 2 D 4 i ' Q c 4 d W b ?  j G ( : c R J N @ L 8 ^ 8 [ u s ? 2 1 ' F S < I A 6 T M a S | 4 b U A 5 / L g v ) 4 X I T B D  u  $  �  �  �  D  f  N  �  o  �  z  �  �      �  �  �  �  �    �  h  �  �  I  �  M  �  �  �  �  �  �       �  q  �  �  i    d  K  �  r  m  �  #  �  #    �  V  |  j  9  b  <  
  �  �  �  ;    �  �  �  �  �  �  d  �  �  �<�o:�o%   �D�����
�o�D��%   ��o��o�8Q켓t���1���-�����aG��t���%���ͽ]/�<j����/��C��q���'�h�\)�,1�C��aG��''�w�u��w�\)�,1�}�8Q�ixսD����h�<j��C����ͽ8Q�Y���C��Y���o�e`B�ixսm�h�u��C���-��7L�u�ixս�C�����
=q��vɽ��w�� Ž�t���t���{���#��vɾ$ݽ�vɽ�Q�����BZ�B4��A��6B,[pB�VB��B�2B��B"�B��B��B#{�B
�B	��B�B C�B/��B w�B.�B��BZB�B�B �B	�B)B"�BQ%B!��A���A�9@B!�RB%b�B��B��B�B�GB$B#=+Bh�B �lB!)�B&��B.B[OB" B
B��B	;B�B	^�B��B'>�Bk�B;�B)8IB�B ~eB��B
�B�B�kBE�B�B��B�?BG�B+�B-�B�`B
9�BG�B�CB�dB= B�BF5B4�LA��B,@�B�.B��B��B@�B"-�B��B.�B#}�B?*B	F�B��B �B/�SB ?SB>�B�B�3BA�B�_B Y�B
4mB��B@BCIB!?�A�̩A�yfB!��B%@
B��BB�B;�B��BH�B#EB��B �B!?B'CB?BG9B� B:�B��B	@{B�TB	H�B0�B&ĝBb�BC'B)@B��B 5~B��B�9B��BF�B��B�TB��B�1B6�B+�
B.FgBJlB
?oB�JB��B�B?�B��A:J�AL�A���@l�B5�B	�A���A�5XAI��B
�
A�%�@��XAƥ�A�&A�FC@�BJAY��A��|A=�.AX?.A���A�/�AJ��B�AZ�qAQ��?�ο@jblAfkA���A�BA+4@���A�S�A��y?�*x@�rg@�"q@K�/B?�A�q�?�F�A+�A��TA�C�l�A�jNAÉ�A�z�A�UwBDJA�<�A�>KA�B	P@��m@���A疥A�i�A���Å_A"EBA�XA��A��4A�ȯAM_(@���A��A���A|A�D'A�5�?/�C���A�^tA:�QALÛA�~�@l�B@tB	��A���A�z(AJ�*B
��Aׁ�@��gA�D0A�+/A�T�@�	�AYu�A�~�A<(�AX��A��vA���AJ�BA�A[ AQ�?��Z@k�AfتA�{�A���A+^@� �A�Z�A�s�?�Y�@�J>@�Ǌ@8��BC�A��?�A+j/A�KAߏ�C�i[A�|AÎ�A�q�A�c�BU"A�y}A���A��5B	R@���@��A�i�A�bA�SA��A'Y�A��TA�A��iA��sAN�@��A�A��A��A�kUA�5�?<��C��A괚         	                        $         >      '      .      $      	      ,   "         	                           k   	               Q   
       ?            
         
   
   	      f               &   N   $         	         2      3      	   &                                    -         +      -      /      +                                             #      /      -            ;         !                              #   -               3   3                     !      %                                                      +      !      %                                                   #            '            -                                          +               3                        !      %            N9/N��cN���N���N��vN7iNJ�'M�?�N&bN5*,O2�YN<[�N��pP&�)N��(O��O5P��N��O�rO ��N�ތNN{�OR��O<��O"�M��$N�X�O�tN�[�O���N�"�N��NE8�O�8�M�hOA�'N�;O��bN�h�O�@�O wHP ~[NLgOCvBO�W}Nkj�NH?�O�O 1O�N���O�ٖN�y�N0�O1�P�RO FiN2�JNbN꤈P6F\O��SOK�O_�O���N�6�N���O=�0O�c�N���O�q�NO EN�^N�^hNx`  �  0  �  �  �  �  '  �  9  5  U  �  �  �  (  R  z  {  �  �  �  \    �  i  P  �  �  �  �  �  %     �  �  �  �  �  <    "  s  �  �  �  
�  �  �  k  3  y  �  .  {  �  �  g  �  8  4  5  �  �  �  n  �  /  �  D  3  5  $  C  S    }<���<t�<o;ě�;ě�;�o;D��;o��o�#�
��9X�e`B�u���㼛��ě���1�+��t���9X��/��j�ě��\)�\)��/������/��`B��h�o�����o�C��+��C��t��\)�\)�t��u��P�#�
�@���w�'D���49X�D���<j�D���H�9�T���e`B��o�]/�]/�]/�]/�ixս�^5�y�#�q����%��o������������T����T�����Ƨ�:<HUX`VUSHA<::::::::��������������������)/;;BF;9/&#&))))))))����������������������
 
��������,/6<AHIHC</,,,,,,,,,##/18//'#����������������������������������������"' �
#/<CHLLH</*#
�����

������������!#/4:5:9/#  !!!!!!AHR[gt���������g[FBA"##/7<<<4/#��������������������*6?CEECA;6*SVaz�����������maYSS��������������������<B[ptyzxtrsh[OD>:8:<������������������#()5BKIBCB5),59BNQNMFB52,,,,,,,,wz������������zyrprwP[gt����������tg_[QP��������������������JOR[hihc[UOLJJJJJJJJ ! ���������������������./6;=@@>;/.+)+......9;HTaluuoa\THD;95459��������������������#/0:;20#").6766)'��������������������ghity{thecgggggggggg�����
�����������36BEOOTROB?6-*333333��������������������������������������������������}}���������������������0<Q\hprqjbUI50)�������������������������
��������������
#/=E?3/$
�������������������������rty�������tsrrrrrrrrNNX[gty~}vtg[YQNMNN��������������������X[gtz����|tqhg[YUSTX�)-21.)"���:CIUbnzzwof_RI<3257:MO[hpsnlmhc[YPOIMMMM��������������������nqv{������������{qon��������������uz������������zrouu�����������������������������������������������������������������)02.)��������
#375/*#
�����	#$/78851/#	����������������������������������������nnz�������}znddfnnnn������������������������
!��������)5NW\fdkh[VN5)qtv���������vtjlqqqq���������������xxz|���������������������zz|���������zzvzzzz<@A?</(#!#/<<<<<<<���������������������4�0�,�/�4�A�M�Q�M�I�A�6�4�4�4�4�4�4�4�4�����������������������žƾ�������������ĳĮĲĳĿ������������Ŀĳĳĳĳĳĳĳĳ������!�-�:�<�A�:�-�!��������I�G�I�J�R�V�b�o�{�~�~�{�p�o�b�V�I�I�I�I�$������$�'�0�6�7�0�$�$�$�$�$�$�$�$�;�9�/�"�"�!�"�/�:�;�H�K�H�?�;�;�;�;�;�;�/�-�#��#�/�<�<�H�R�H�<�/�/�/�/�/�/�/�/�����������������������������������������0�0�0�0�7�=�I�K�K�I�@�=�0�0�0�0�0�0�0�0����� �%�)�/�6�B�I�Z�S�O�K�B�6�1�)������������������������������������������H�C�A�H�U�a�n�zÇÆ�z�n�a�U�H�H�H�H�H�H�H�/�����������"�;�a�m�u�t�����m�a�HàÙÓÓÓ×àëìíù��ÿùðìàààà�Y�@�4�'����$�4�<�M�Y�f�r��������r�Y�	�������������	���$�%�"���	�"�����	���"�/�H�T�[�\�X�T�S�V�T�H�"�A�;�4�4�4�5�A�M�Z�`�f�i�p�f�Z�M�A�A�A�A�������׾޾���	��"�.�4�3�4�.�"����ʾ��Z�T�N�K�L�N�S�Z�g�s������������s�g�Z�Z�Z�Y�N�A�@�:�9�A�N�Z�g�j�i�o�o�g�Z�Z�Z�Z��������������������������������������������������������������#�!����������������������	���"�"�$�!����	�����������������ʾ׾�����������׾ʾ����Y�W�Y�]�d�e�j�r�t�r�i�e�Y�Y�Y�Y�Y�Y�Y�Y��������!�-�7�4�-�*�!�������;�6�.�*�*�.�4�;�G�T�`�m�s�w�m�b�`�T�G�;�����������������������������������������������(�5�A�U�Z�h�t�s�c�Z�A�5�(��ݽѽнŽȽнݽ����������ݽݽݽݽݽݻû����ûŻлܻ����ܻл˻ûûûûû�ìçéì÷ù��������ÿùìììììììì��������������*�6�C�O�\�h�O�C�6���������'�,�)�'������������x�w�p�r�x�����������������������������x�_�]�]�_�k�l�x���������}�x�l�_�_�_�_�_�_�ɺ������������ɺ���!�1�-�(�!�����ƧơƣƧƳ��������ƳƨƧƧƧƧƧƧƧƧƧÓÍÇÃÇÏÓë����������	�������àÓ�L�K�K�N�R�Y�b�e�r�|�~���������~�r�e�Y�L�������������нݾ�(�6�2�!�ݽнĽ��������)�&�)�1�6�B�O�Q�O�B�=�6�)�)�)�)�)�)�)�)�h�`�qāčěĦĳĻĿ����ĿĵĦĚčā�t�hD�D�D�D�D�D�EEEEE*E,E7EDEDE:E*EED�Ň�}ŀŇőŔŗŠŢťŠŔŇŇŇŇŇŇŇŇ�<�5�/�(�+�/�<�C�H�Q�J�H�<�<�<�<�<�<�<�<����������������)�,�3�,�)�����ŭŪŠŔŎŔşŠŭŹ��������������Źŭŭ�6�3�2�3�6�>�C�O�Q�\�f�h�u�v�l�h�\�O�C�6�T�P�H�H�F�D�H�T�a�m�u�z�|�z�t�m�l�a�T�T���s�n�h�d�f�s������������������������������������������������������������������������$�&�+�$��������������������������������ûлܻ���ۻлû��f�M�C�=�<�B�M�Y�r���Լ��ݼѼ������r�f�����������������
������
����������0�-�%�0�<�G�I�K�P�I�A�<�0�0�0�0�0�0�0�0������������������������ààÖÕÓÓÓàìôù����������ùìàà�y�`�G�6�;�`���������������ݽн��y�ݿֿѿ˿ǿǿѿݿ�����)�#�������������s�g�b�e�g�q�����������������������������������(�5�6�<�:�7�5�(���H�;�2�2�7�;�?�H�T�a�j�m�w�������z�Y�T�H���������������������ʾ̾ξʾ������������r�o�r�x�����������������������r�r�r�r�����ļʼ׼���������������ּ��z�n�b�]�b�nÇÓàìùþ��������ìàÓ�z�#�����#�$�/�<�H�J�I�H�H�<�/�#�#�#�#�z�o�m�n�u²����������������¿²ā�}�t�q�tāčĚĥĚđčāāāāāāāā�ܹ۹ܹݹ���������������ܹܹܹ�D{D�D�D�D�D�D�D�D�D�D{DpDqD{D{D{D{D{D{D{�������#�0�5�8�0�#�������� W $ 0 D < 0 Q � ] Z i T e $ * C 2 P 4 ?  Q G + V U b ?  j : ( : c C J ' @ E 8 ^ 8 h u c 3 2 +  F 0 < I A 6 6 G a S | 4 b ( ? 5 / L g v ) 4 R I T B D  u  �  �  �  �  D  f  N  S  o  �  z  �  �  �  a  D  �  �  �  T    y  �  �  �  I  �  M  �  1  �  �  �  �     �  �  �  �  �  i  �  d  �  S  r  U  B  #  A  #    �  V  w  �  9  b  <  
  �    �  ;    �  �  �  �  �  E  d  �  �  �  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      $  )  ,  /  /  .  *  #      �  �  �  �  _  -   �   �  �  �  �  }  p  d  W  J  :  *      �  �  �  �  �  r  Y  @  �  �  �  �  �  �  �  �  �  h  M  /    �  �  �  \  )   �   �  �  �  �  �  �  �  k  P  4    �  �  �  �  y  A    �  �  N  �  �  �  s  ]  J  6  $       �  �  �  �  �  �  �  d  9    '    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  m  ^  P  B  4    �  �  �  �  t  R  0     �  "  '  ,  1  6  9  7  6  4  2  1  0  .  -  ,    �  �  �  �  5  /  *  %    �  �  �  �  v  R  .  
  �  �  �  u  N  '     #  �  �  �    0  R  R  G  1  ?  #    �  �  �  M  �  /  �  �  �  �  �  �  �  �  z  k  \  L  :  )      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  n  �  �  �  �  �  �  �  x  c  C    �  �  |  #  �  i  %  �  �  H  �    �  �      #  &  '  #          �  �  �  �  �  c    �  �  %  A  N  R  H  3  5  /  (    (  /  '    �  �  �  W  �  x  w  s  x  z  u  k  \  L  8  #    �  �  �  z  :  �  �  i  Y  g  e  b  c  k  x  t  ]  >    �  �  l    �    |  �   f  �  �  �  �  y  h  V  C  1    
  �  �  �  �  �  �  �  t  a  <    �  ~  v  q  j  a  \  T  H  0    �  �  >  �  �  a  F  �  �  �  �  �  �  �  �  �  �  r  M    �  �  p  .  �  �  O  \  N  @  <  =  <  7  1  .  ,  #      �  �  �  z  G     �  �             �  �  �  �  �  �  �  �  �  �  �  w  f  V  w  �  �  �  �  �  �  �  �  �  k  /  �  �    �  3  �    )  �  �  $  I  `  h  f  `  W  J  9  &    �  �  <  �     ?  K  O  P  L  B  5  ,    �  �  �  }  W  0    �  �  �  A  �  v  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  g  Y  K  <  �  �  �  �  �  �  �  �  �    }  �  �  v  h  X  G  5  "    �  �  �  �  �  �  x  _  ?    �  �  �  q  C    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  e  R  >  �  �  �  �  �  �  �  y  k  V  <    �  �  �  n    �  ]  �  %        �  �  �  �  �  �  �  i  J  )    �  �  �  N               �  �  �  �  �  �  �  �  �  �  �  q  W  <     �  �  �  �  �  z  l  ]  N  ?  /    	  �  �  �  �  T    �  q  �  �  z  j  \  M  @  G  4      �  �  �  l    �  �  `  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	&  	�  
-  
�  �  -  q  �  �  �  �  d    �  
  
C  	-  �  �  �  �  �  �  �  �  �  �  �  k  R  7    �  �  �  �  t  [  \  \  7  6  :  +      �  �  �  �  �  �  �  }  p  U    �  �  T    �  �  �  �  �  �  �  u  e  O  3    �  �  �  H    �  �  "  �  �  �  u  c  `  s  �  �  o  D    �  ~  4  �  �  �  �  s  X  ?  (    �  �  �  �  |  _  D  3  "    �  �  �  �  n  �  "  Q  ~  �  �  �  �  �  Y    �  �  ?    �  {  �  )    �  �     �  �  �  e  8    �  �  ~  L    �  �  z  C     �  �  �  �  �  |  9  �  �  �  �  �  j  4  �  �  G  �  }  l  �  
2  
Z  
�  
�  
p  
S  
<  
$  	�  	�  	m  	
  �    g  �  �  �  �    �  �  �  �  u  j  ^  Q  B  4  %      �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  B    �  �  {  E    �  �  X    �  
  0  P  c  j  b  Q  =  &    �  �  �  v  @  �  �    A  F  3  &      �  �  �  �  �  r  P  /    �  �  �  �  �  �  �  N  N  G  y  v  r  f  W  C  *    �  �  �  �  �  �  t  <  �  �  �  �  �  �  �  �  `  ?    �  �  �  �  \  1      �  �  .  "    
  �  �  �  �  �  �  �  �  c  E  &     �   �   �   �  {  j  Y  H  7  0  +  !      �  �  �  �  �  y  @    �  �  �  �  �       
      �  �  �  �  �  �  a  :    �  �  X  |  u  x  �  �  �  �  |  k  T  9    �  �  �  �  [  *   �   �    [  g  W    
�        
�  
�  
a  
<  	�  	  X  v  Q  g  8  �  �  �  �  �  �  k  @    �  �    Z  9  !    �  �  �  �  8  7  6  5  /  '         �  �  �  �  �  �  �  �  Q    �  4  .  )  #        	  �  �  �  �  �  �  �  �  �  l  U  ?  5  +          �  �  �  �  �  h  R  ?  .      �  �  �  �  �  d  =    �  �  �  ^  *  �  �  �  �  T  #  �  �  q  �    i  `  W  J  *  �  t  �  �  �  �  x  ?  �  `  �    B  T  V  �  �  �  �  �  �    T  $  �  �  q  $  �  k     �  �  �  �  n  M  +  
  �  �  �  �  ]  5    �  �  x  @    �  F  �  �  �  �  �  �  �  f  D     �  �  �  ]  ,  �  �  �  S  �  �    /  !      �  �  �  �  �  �  �  �  �  m  X  7    �  �  �  �  �  �  �  �  p  ]  J  6    �  �  �  �  �  K     �   �   f  D  (    !  �  �  �  �  `  .  �  �  �  �  `  7  �  n  �  (  3  $    �  �  �  �  b  0    �  �  �  T    �      �  �  5  &       �  �  �  �  a  =    �  �  �  z  J    �  x  !      �  �  �  �  �  �  S    �  �  ~  (  �  6  �    W  }  C    �  �  �  Q    �  �  O    �  s  '  �  �  >  �  �  .  S  Q  O  J  B  9  -  "      �  �  �  �  �  z  [  +  �  �    �  �  `  +  �  �  X  
�  
�  
!  	�  	  E  y  �  �  �  �  �  }  w  q  k  e  _  U  L  B  9  -         �  �  �  �  �  �