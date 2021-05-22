CDF       
      obs    B   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�t�j~��       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�@   max       P�K�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��1   max       <��
       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @F�z�G�     
P   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v�G�z�     
P  +   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @P            �  5d   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�Y�           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �I�   max       <e`B       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B5�       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�}�   max       B4�-       9    	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�Qc   max       C��J       :   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�h   max       C��m       ;   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          M       <   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?       =    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9       >(   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�@   max       P�~�       ?0   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�U2a|�   max       ?�ѷX�       @8   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��1   max       <���       A@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=q   max       @F�z�G�     
P  BH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v�G�z�     
P  L�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @P            �  V�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�           Wl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @0   max         @0       Xt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ᰉ�(   max       ?Ծ�(��     �  Y|      
                  D      
   
      2         <               	      1   =      0                                                    #      	               L   	   3   
                   <                     N�3N���Nd-BO���N��dO}KO̵P�K�N���OG�O��OU�P��vO�CaOha:Pn�N�#�O�UOZ*N��ND!8N��MP5�LP!�N뙝O�NN_'zN9��Oh�N�@N�bO�mNF��PY�O<��N���Oq��Nb;�O5�0O%�N=MO�ݡO���O��N�|�O@�IPF�N���N�XP#@N�z�P��N�[�O��OH*FO�r\N�D�N�m�P�NK��N�F�O�)N�b�O�>�O/��N�.6<��
<u<D��;ě�;o�o�D����o�ě���`B�t��t��D���e`B�e`B��o��t���t���t���t���1��1��9X��9X��j��j��j�ě����ͼ�����h����P�#�
�#�
�',1�0 Ž49X�8Q�D���H�9�L�ͽY��aG��aG��aG��ixսq���y�#�y�#��%��%��������hs��hs���㽛�㽛�㽝�-���-���
���
���1�������������������������������������������������������������������������+07<IU[XUQKI<50.++++"/;BGIB;:2/"")/35BFDNB?BEB5#�����������~��UH=</-$#!#'/3<AHSUWUAINW[g{�����tg[NGB@A����������������������!#$#
�������)6[hx{���zqO6*/4?IUVanxnUH/#����!('! ���������������������T[]]gktz�������tg[TT�������������������
/<UHC<6/
����36BGOOTW[_[OBA=76133��������������������R\hu������vuh`\URRRRTam���������zmaZTMNT�	#/HUaf_WH<8
�����|������������|wx||||��
#/:HTLD</$
�����SUadfecaYUTQQRSSSSSSpt�����tmnpppppppppp9<HUagga``XUHC<95599��������������������6BOW[b[[TOMBB6666666��������������������_amtyz}zmha`________)BQ[knjc^\YNB:BIOU[hmtqmllh][OB>;B��

��� ����������������������������������������������mnsz����������zzumkm����������������������������������������=DN[t���������tgZF<=nuz�����������zmddgn��������������������mt~����������{ttmmmm����
#(08;9/#
����>Xjn{|������{ncUNIB>//:<HSUXVUH<0///////����������������������)45:8-&"��������������������������D[g�������������tgND��������������`htv{|zzxthga]\]__a`�����������������������������������������������������v���������������vvvv#,/@BHQahliaUH</##HHNUaleaYUHFHHHHHHHH���� 

������������� ## ��������"#/<<=<:/#""""""""HOUan{������naUHFACH,7>FIJPTTVVTPI<3,'),#)*5BEIGDB<5)($"####�������������"��������������������n�h�c�n�zÇÓàáàÕÓÇ�z�n�n�n�n�n�n�нʽĽ����Ľнݽ߽��ݽннннннн��#���
���������
��#�/�<�@�C�H�<�5�/�#�ʼż����������ʼϼּ����ټּʼʼʼ���	���������	��"�C�H�T�X�_�T�H�;�/�"�����������������������%�1�)�"���y�`�R�C�?�N�`�y�����ʿ������"�(���Ŀy���������������������������������������������������������������������������������T�M�G�A�?�G�T�`�m�y�~���������y�m�`�T�TÝÖÝàìðùû������������������ùìÝ�����.�����.�G�`�m�������Ŀֿ׿пĿ��<�/����#�/�9�<�H�J�U�^�k�o�g�a�U�H�<�N�A�5�1�)�+�5�A�N�Z�g�p�s�y�|�y�s�i�g�N�����������������������H�Q�a�k�o�h�H����|�z�m�l�m�q�m�j�m�t�z�����������������s�p�g�_�Z�N�A�?�A�D�N�Z�k�s�~���������s�����+�1�B�O�[�b�[�S�O�K�G�B�9�6�)��S�O�Q�S�]�_�l�x�������������x�l�j�_�S�SFFFFF$F1F=FJFQFJF=F1F(F$FFFFFF�G�@�?�@�@�D�G�T�[�W�`�a�g�`�\�T�G�G�G�G�S�,�!�'�5�A�N�g�s������������������s�S�4�(�-�-�0�4�A�M�f�s�z�|���������f�Z�4�����������Ŀѿݿ�����ݿѿĿ�������D�D�D�D�D�D�D�D�D�EEEEE D�D�D�D�D�DӾ��������	��"�&�.�0�.�"��	�������������H�B�E�H�U�a�d�f�a�U�H�H�H�H�H�H�H�H�H�H�g�d�^�^�f�g�s�|�������������������s�g�g�����������������������������������������a�X�\�a�g�n�z�zÇÐÇ�z�z�n�a�a�a�a�a�a�M�O�Z�l�s�������������ʾҾ־����s�f�Z�MčĉćčĚĝĦİĬĦĜĚčččččččč����������Ž����������*�6�D�F�C�%����ֹù����������ùϹܹ���������������������$�0�8�9�6�0�(�$������������������������������
����������̾��������޾��������	��	�������������������������������ƾʾ;˾ʾ�����������������u����������ʼͼʼż����������� ������$�%�'�$����������������������	��#�.�7�2�)�"��	��ƎƁ�z�t�{ƁƎƚƳ����������������ƳƧƎ�������������ûлӻܻݻ߻�ܻܻлû��������������������������������������������ػۻӻлλ����������ûлܻ�������ݻۻл���������������@�Y�^�]�T�@����ܻп�����������������������������������������������Ŀľ�����������������������������õåÍÇ�w�zÓù������+�0�)���������r�k�e�_�e�p�r�~���������������������~�rā�s�e�7�%� �6�B�O�[�h�v�xĈĒĒěĜĕā¦²µ³²®­¦�����������������	��"�"��	�������������ּּݼ������!�.�1�.�+�!�������ֺ��~�y�v��������ɺ�����!���ֺ������F�:�:�-�!�-�:�F�S�_�l�x���������x�l�S�F�<�7�6�;�<�H�Q�U�W�a�b�a�V�U�H�@�<�<�<�<E�E�E�E�E�E�E�FF1FVFcFoFqFgFXF@FE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����� ����������!�-�.�.�/�.�!����O�=�G�R�������������ݽнĽ������y�`�O�����������������������������������������^�T�P�J�N�T�_�a�z�����������������z�m�^����ݽн˽нݽ������(�2�3�(����������������������
������
���������� 1 Y ? 4 = M _ > ^ = G ' ? Q  4 b f v i t w 5 ; ! h g 2  6 B b T < A [   � ) > E M * C D W ] , = q 6 V = t Z f y h q X C u 0 3 H .  �  �  x    �  !  ?  '    �  8  �  h  d  �  �    |    �  �  D    Y      �  J  1  '  �  �  n  �  �  �  �  �  �  �  [  �  S  =  �  �  !  �    l     �  �  �  �  `  �  �  Q  a    =  �  I  �  �<e`B;��
;��
��t��#�
��`B�49X��hs��1��C���t����}�t��o������9X��h��w�������ͽ�\)��C���hs�������49X��`B�49X�H�9�49X��hs��7L�D���ixս@��q����C��T�����-��1��t���%��C��� Ž����7L�I���\)��xս�t����㽺^5��񪽮{��Q�C���1��{���`�� Ž��`���ͽ���B1YB��B!��B��B&��A��B�B+b�B�B	$UB<TB�uB4pB��B��B�BB	�9BZBB��Br�B�B2�KA��BB
��B`B��B�CBiLB5�B��B �A�t�B�[B4�B�VB/�B|�B�7BUB,B	�?B a�B ��B
W2B$�LB(�MB
B^0B=sB!�B
�B1�B�B-��B��By+B
��B?�B܄B��B��B$~B�B&y=BFB@ B��B!̇B�B&�A�}�BѡB+>�BY7B	:�B<BC)B��B@�B�3BC'B	]6B��B�FB?�B?�B2S�A��BW�B
�=B@�B�oB��BA4B4�-B�BB A;A��B�_B?yB�9B?1B��B�/B&B@�B	��B ?�B �1B
A1B$�NB(��B;BP(B=�B!�sB
��B>>B��B.@�B�]B@B
�BU�B> B�0B�8B9B��B&��B@�B��A�(�A)bCA�6@��'A�(A��Au| A��.A��Ai�A��$Al��A�MdA�ݦA�H�A�)A�e,A׈@��C��JAe7KA�RA?5�AzJeC�2�A\�KAŉ�A�)]AJ�fA�8`AGSzA߄vA�3>�QcB	�*A��eAXfAK_�@�B	�AZ�B�@���A�v@���@���Asd�A��iA��@@W�Aڠ�A���A��A�-@2f�@�J�AĩpC��C�OA
�ZA 4�A���A�NA0��A�g�B	Aɍ�A(��A���@��A��A�u`Apd�A�UA��LAjb�A��AAo`A�>�A�m�A�A��vA��A׀�@��)C��PAf��A�A?�nAz�C�0 A]r�AŁ�A���AK[vA�~�ACj�A�j�A��">�hB	��A���AV�?AIkG@��B	AZ�&Bo�@�)+A�C@�d�@��zAsA�A��~@;&A���A��cA�ZA
�@%��@� 1A�{�C��mC�AA�A!��A��A�}-A4��A���                     	   D         
      3         <               
      1   =      1                                                    #      	             	   M   
   4   
         !         =   	   	                                       ?               9   !      3                     )   %                        !      '                                       -         /      +            %         -         '                                    9               /   !                           !                                 '                                       -         !                  %         +         '            NE_�N���Nd-BO���N��dO$�N��lP�~�N���OG�O��O0��PGg�O�CaOha:OzՠN�#�O�UOZ*N��ND!8N��MOԣcO�3NͬGO��[N_'zN9��O 82N�@N$�Oy��NF��PY�N��N=UOq��Nb;�O5�0O%�N=MO�O���N�:�NM$BO@�IP��N���NQ;�O��N�z�O��N�[�O��OH*FO�r\N���N�m�PD�NK��N�F�O�Z�N�b�O�>�O/��N�.6  �  �  !  �  �  �    �  a  E  }    �  /    I  l  l  �  �  t  T  X  �  -  	C  �  ^  �  ?  �  �  �  �  �    �  �  y  j  �  �  �    �  �  A  u  �  �  �  f  G  �  �  /  
    �  ]  �  +  
  �  �  :<���<u<D��;ě�;o�o���
�T���ě���`B�t��D�����
�e`B�e`B�<j��t���t���t���t���1��1��P�t��ě����ͼ�j�ě����������C��o��P�#�
�<j�,1�,1�0 Ž49X�8Q�D���q���L�ͽ]/�e`B�aG��ixսixս}󶽙���y�#���P��%��������hs������㽝�-���㽝�-���w���
���
���1��	�����������������������������������������������������������������������+07<IU[XUQKI<50.++++"/;>@<;;95/#"%)457<<@5)%%%%%%%%�������
����������UH=</-$#!#'/3<AHSUWUAINW[g{�����tg[NGB@A�����������������������
##
�������" )6Bflkp{|zpeOB6*/4?IUVanxnUH/#����!('! �������������������������T[]]gktz�������tg[TT�������������������
/<UHC<6/
����36BGOOTW[_[OBA=76133��������������������R\hu������vuh`\URRRRTXamz��������zmaVRRT� 
/HUY^UM<)
����������������}yz��
#/7<IMIB</#
�����SUadfecaYUTQQRSSSSSSpt�����tmnpppppppppp:<HU`afa__VUHE<966::��������������������;BOS[][SOKB<;;;;;;;;��������������������_amtyz}zmha`________)BQ[knjc^\YNB:BDHLOZ[ghpmjihg[ONBB��

�������������������������������������������������mnsz����������zzumkm����������������������������������������LN[agt�������trg[ULLnuz�����������zmddgn��������������������st�������}utssssssss����
#(08;9/#
����E\n{��������{ne]WQIE//:<HSUXVUH<0///////������������������������&"!�������������������������tx�������������tlhlt��������������`htv{|zzxthga]\]__a`�����������������������������������������������������v���������������vvvv#-/<HPahkkiaUH</##HHNUaleaYUHFHHHHHHHH���� 

�������������  #"�������"#/<<=<:/#""""""""HOUan{������naUHFACH,7>FIJPTTVVTPI<3,'),#)*5BEIGDB<5)($"####�������������!������������������n�h�c�n�zÇÓàáàÕÓÇ�z�n�n�n�n�n�n�нʽĽ����Ľнݽ߽��ݽннннннн��#���
���������
��#�/�<�@�C�H�<�5�/�#�ʼż����������ʼϼּ����ټּʼʼʼ��	����	��"�.�/�;�H�Q�T�T�H�;�/�"��	�������������(�(�������������y�Y�O�E�J�T�y�����ѿ���������ѿ����������������������������������������������������������������������������������T�M�G�A�?�G�T�`�m�y�~���������y�m�`�T�TìâÜÞàáì÷ù������������������ùì�����;�!���.�;�G�T�y�������ĿϿѿϿĿ��<�/����#�/�9�<�H�J�U�^�k�o�g�a�U�H�<�N�A�5�1�)�+�5�A�N�Z�g�p�s�y�|�y�s�i�g�N�/�(�"�����"�/�;�H�T�X�_�`�W�T�H�;�/���|�z�m�l�m�q�m�j�m�t�z�����������������s�p�g�_�Z�N�A�?�A�D�N�Z�k�s�~���������s�����+�1�B�O�[�b�[�S�O�K�G�B�9�6�)��S�O�Q�S�]�_�l�x�������������x�l�j�_�S�SFFFFF$F1F=FJFQFJF=F1F(F$FFFFFF�G�@�?�@�@�D�G�T�[�W�`�a�g�`�\�T�G�G�G�G�f�Z�<�.�1�9�A�Z�s�������������������s�f�Z�M�A�8�4�2�5�=�A�M�a�f�o�������s�f�Z�����������Ŀѿݿ�����ݿѿĿ�������D�D�D�D�D�D�D�D�D�D�EE
EED�D�D�D�D�DӾ��������	��"�&�.�0�.�"��	�������������H�B�E�H�U�a�d�f�a�U�H�H�H�H�H�H�H�H�H�H�g�e�^�_�g�j�s�y�������������������s�g�g�����������������������������������������a�\�_�a�l�n�q�z��z�t�n�a�a�a�a�a�a�a�a�Z�P�Z�f�n�s�����������̾о��������s�f�ZčĉćčĚĝĦİĬĦĜĚčččččččč����������Ž����������*�6�D�F�C�%�����ܹϹù����������ùϹܹ�������������0�*�$�����$�0�6�7�4�0�0�0�0�0�0�0�0������������������������
����������̾��������޾��������	��	�������������������������������ƾʾ;˾ʾ�����������������u����������ʼͼʼż����������� ������$�%�'�$�����������������������	���#�#�"����	��ƎƁ�z�t�{ƁƎƚƳ����������������ƳƧƎ�����������������ûлܻ޻߻ܻڻлû��������������������������������������������ػۻӻлλ����������ûлܻ�������ݻۻл�����������'�@�M�W�\�[�Q�@�'���ֻп�����������������������������������������������������������������������������������þùðíëìù��������� ������Һr�k�e�_�e�p�r�~���������������������~�r�Y�O�=�*�)�:�C�O�[�h�j�xāćĈćā�t�h�Y¦²µ³²®­¦�����������������	��"�"��	�������������ּּݼ������!�.�1�.�+�!�������ֺ��~�y�v��������ɺ�����!���ֺ������F�?�:�F�S�_�l�x���������x�l�_�S�F�F�F�F�<�7�6�;�<�H�Q�U�W�a�b�a�V�U�H�@�<�<�<�<E�E�E�E�E�E�FF1FJFVFcFnFqFgFXF@FE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����� ����������!�-�.�.�/�.�!����`�S�A�T�����������߽��ݽнĽ������y�`�����������������������������������������^�T�P�J�N�T�_�a�z�����������������z�m�^����ݽн˽нݽ������(�2�3�(����������������������
������
���������� 9 Y ? 4 = R H 9 ^ = G   ? Q   b f v i t w : C % e g 2  6 I V T < ; O   � ) > E H * ; 6 W ` , E = 6 9 = t Z f n h p X C x 0 3 H .  Z  �  x    �  |  �  L    �  8  v  �  d  �  �    |    �  �  D  �  �  �    �  J    '  Y  6  n  �  %  s  �  �  �  �  [  o  S  "  j  �  �  �  {  �     O  �  �  �  `    �  6  a    �  �  I  �  �  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  @0  q  w  |  �  y  n  b  M  4      �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  g  Q  9    
     �  �  �  �  ^    �  #  !    	  �  �  �  �  �  �  �  e  C    �  �  �  �  a  8    �  �  �  �  �  �  �  x  Z  6  
  �  �  g  $  �  O  �  -  R  �  �  �  �  �  �  �  �  �  q  X  <    �  �  �  �  R    �  5  W  s  �  �  �  �  x  f  K  ,    �  �  �  �  �    �  6  �  �  �  �           �  �  �  �  �  �  x  O  "    +  F  E  �  �  �  �  Y    �  �  �  �  \  '  �  �  h    �  (  2  a  D  !  �  �  �  |  M    �  �  {  I    �  �  �  �  \  3  E  7  )    
    �  �  �  �  �  �  �  �  �  �  t  F     �  }  {  y  q  i  X  D  ,    �  �  �  �  e  =    �  �  w  :  	          �  �  �  w  W  4        �  �  �  �  ^  +  �  �  �  �  �  �  �  �  �  �  O  #  �  �  �  ;  �  �  �   �  /    �  �  �  �  j  M  /  '  )  "    /  2  %    �  �  !    �  �  �  �  �  �  �  �  �  w  V  1  	  �  �  j  H  �  z  *  �  �  =  ?         2  F  E  '  �  �  �  P  �  :  �    l  _  R  E  8  +        �  �  �  �  �  �  �  �  �  �  �  l  l  j  c  Y  J  4      �  �  �  d  9    �  �  S     �  �  �  �  �  j  D       �  �  �  �  X    �  �  �    �  �  �  �  �  �  �  �  �  �  k  F  !  �  �  �  �  \       �  �  t  m  g  �  �  �  �  �  �  f  J    �  I    �  �  N     �  T  C  2  !    �  �  �  �  �  �  �  r  b  Q  6     �   �   �  �  �  *  A  M  T  W  O  8    �  e    �  ,  �    |  �  b  A  i  �  �  �  �  �  m  7  �  �  �  B  �  �    b  �  �  �  *  ,  -  +  '  "      �  �  �  �  l  >    �  �  Y     �  	"  	@  	)  	  �  �  �  �  {  u  !  �  6  �  �  x  )  �  }    �  �  |  y  v  r  o  l  i  f  c  a  ^  a  r  �  �  �  �  �  ^  U  K  B  8  /  %        �  �  �  �  �  }  e  J  0    �  �  �  �  �  �  �  �  k  R  7    �  �  1  �  �  !  �  �  ?  <  9  7  4  1  /  ,  )  '  $  "                   <  `  {  �  �  �  �  �  �  �  �  n  Q  +  �  Y  �  �    �  �  �  �  z  l  o  ^  A  '      �  �  �  �  k  .  �  �   �  �  �  �  �  �  �  �  �  �  o  P  1    �  �  �  �  k  J  )  �  {  u  i  \  a  o  n  a  F  '    �  �  �  K     �  �  �  Z  k  ~  �  �  �  �  g  >    �  ~  W  �  �  n  -  �  J  �              �  �  �  �  �  �  �  �  �  �  �  �  �  g  �  i  W  I  >  6  4  2  0  -  /  ,  $    	  �  �  �  �  \  �  �  |  u  n  d  P  =  *      �  �  �  �  �  H   �   �   k  y  i  V  >  !    �  �  �  _  1    �  �  o  4  ?  N  Q  U  j  [  L  ?  <  H  G  B  4       �  �  �  y  A    �  �     �  �  �  �  �  �  �  �  }  t  l  d  \  T  L  5    �  �  �  j  �  �  �  �  �  �  �  �  �  �  q  N  %  �  �  B  �  ;  �  �  �  �  �  ~  o  Y  A  %  �  �  �  `  $  �  �  5  �  �  �  �    �  �  �  s  6  �  �  X    =  O  W  ,  �  �  =  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  f  P  :  +  '  #  �  �  �  �  �  t  c  L  2      �  �  �  �  �  s  U  N  b  *  @  ;  "    �  �  p  W  r  I  '    �  �  �  w  <  �  �  u  U  @  $    �  �  �  �  x  U  .     �  r  	  �  
  �   �  o  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  W    �  [  
�  ;    �  �  h  ,  
�  
�  
2  	�  	Y  	  �  H  �    1  
  �  �  �  �  �  �  �  �  �  �  k  R  9       �  �  �  �    L  X  '  <  `  f  d  Q  A    �  �  C    �  �    �  �  �  �  G  B  >  4  )        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  `  K  8  &      �  �  �  �  �  �  �  �  �  �  �  �  �  }  [  A    �  �  �  t  G    �  �  U  �  �  �  J  /      �  �  �  �  X  )  �  �  \    �  F  �  �  U    �  �  �  �    �  �  �  �  �  �  b  @      �  �  �    =  �    �  �  �  �  �  �  t  U  5    �  �  �  �  �  j  /  �  �  �  �  u  &  �  _  
�  
l  	�  	�  	b  �  �    r  �  �  �    E  ]  ]  ]  R  @  .      �  �  �  �  �  �  "  %        �  �  �  �  �  �  h  O  6      �  �  �  �  �  �  �  Q  �  f  *  +  '    
  �  �  �  ~  R  $  �  �  l    �  �  �      
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  c  Q  ?  �  �  �  �  �  �  �  �  {  d  J  -    �  �  �  p  %  ]   �  �  �  �  j  H  &    	    �  �  �  �  `  +  �  �  }  L  ~  :  "    �  �  �  �  k  A    �  �  9  �  c  �      �  �